#!/usr/bin/env python3
"""Policy server for the joint-impedance (manipulation) controller — ROBOT PC.

Same pattern + zmq port as franka_server / calib_server (mutually exclusive: run
whichever the current mode needs). Runs JointImpedanceController (1 kHz torque
loop) and streams 21-D actions from the perception PC's policy.

Protocol (zmq REQ/REP):
  {"cmd":"ping"}                 -> {"ok":true}
  {"cmd":"get_state"}            -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4],"jacobian":[[7]x6],"controlling":bool}
  {"cmd":"set_action","a":[21]}  -> {"ok":true}      (a = [Δq(7), Kp(7), Kd(7)])
  {"cmd":"gripper","action":"close"|"open"[,"width","force"]} -> {"ok":true,"width","is_grasped"}
  {"cmd":"reset","gripper":"close"|"open"|"none"} -> {"ok":true}  (episode reset: stop -> home -> re-arm)

On startup it parks at HOME and closes the gripper, then holds at home under impedance
(--no-home / --no-gripper to skip; --home-gripper to calibrate the gripper first).
Run on the ROBOT PC, e-stop in hand — the arm becomes live on startup:
  python servers/impedance_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556 --max-dq 0.5
"""
import argparse
import gc
import os
import sys
import time
import traceback

import numpy as np
import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.joint_impedance_controller import JointImpedanceController, DEFAULT_KP
from controllers.gripper_service import GripperService
from controllers.home import Q_HOME, go_home, set_gripper


def _impedance_lift(ctrl, dz, hz=10.0, step=0.03, tol=0.01, timeout=8.0, lam=0.1):
    """Retract the EE straight UP by `dz` m (base +z, orientation HELD) THROUGH the impedance controller, BEFORE
    homing. A direct joint ramp to home sweeps the arm through the bin and topples the just-reoriented object;
    lifting clear of the scene first avoids that (and disengages from a timeout-on-contact). Jacobian damped-
    least-squares turns the Cartesian up-step into Δq: dq = Jᵀ(JJᵀ+λ²I)⁻¹·twist, twist=[0,0,dz-step,0,0,0].
    Returns True once it has cleared dz, False on timeout. No-op if dz<=0."""
    if dz <= 0:
        return True
    z_target = float(np.asarray(ctrl.get_state()["T_base_ee"])[2, 3]) + dz
    t0 = time.monotonic()
    while True:
        st = ctrl.get_state()
        z = float(np.asarray(st["T_base_ee"])[2, 3])
        if z_target - z < tol:
            return True
        if time.monotonic() - t0 > timeout:
            return False
        J = np.asarray(st["jacobian"], float)                # (6,7) base-frame zero-Jacobian
        twist = np.zeros(6); twist[2] = min(step, z_target - z)   # +z only, no rotation -> pure vertical retract
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), twist)
        ctrl.set_action(np.concatenate([dq, DEFAULT_KP, np.full(7, 2.0)]))   # set_action clamps Δq to max_dq
        time.sleep(1.0 / hz)


def _impedance_goto(ctrl, q_home, hz=10.0, step=0.1, tol=0.03, timeout=15.0):
    """Ramp the arm to q_home THROUGH the running impedance controller (compliant torque, loose collision),
    NOT the position controller. A reorient that TIMED OUT parked on the object leaves the arm loaded; position
    control re-trips the Reflex at configure (its low collision thresholds vs the residual contact force), but
    the impedance loop just holds + eases off compliantly. Streams a per-tick Δq toward home (bounded to `step`
    rad, ~= the golden max-dq, so it's gentle even if the server runs without --max-dq) at the policy's ~10 Hz
    cadence with critically-damped default gains, until reached. Returns True if it arrived, False on timeout."""
    q_home = np.asarray(q_home, dtype=float)
    kd = np.full(7, 2.0)                          # Kd COEFFICIENT (set_action multiplies by sqrt(Kp)); 2.0 ~= critical
    dt = 1.0 / hz
    t0 = time.monotonic()
    while True:
        q = np.asarray(ctrl.get_state()["q"], dtype=float)
        err = q_home - q
        if float(np.max(np.abs(err))) < tol:
            return True
        if time.monotonic() - t0 > timeout:
            return False
        d = np.clip(err, -step, step)            # bounded per-tick target advance -> gentle, server-max_dq-independent
        ctrl.set_action(np.concatenate([d, DEFAULT_KP, kd]))
        time.sleep(dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-delta-tau", type=float, default=0.8,
                    help="per-tick torque slew limit [Nm]; at 1 kHz this caps dτ/dt (0.8 -> 800 Nm/s, under "
                         "the ~1000 Nm/s reflex). Final backstop — gains+target are interpolated so it rarely binds.")
    ap.add_argument("--max-dq", type=float, default=0.5, help="safety clamp on |Δq| per action [rad]")
    ap.add_argument("--interp-time", type=float, default=0.1,
                    help="linear target-interpolation ramp time [s] ~ the policy period (10 Hz -> 0.1). "
                         "Ramps q_ref to the new target so the 1 kHz loop tracks a smooth ramp, not a step.")
    ap.add_argument("--reference-mode", choices=["measured", "commanded"], default="measured")
    ap.add_argument("--home", type=float, nargs=7, default=Q_HOME, metavar="Q",
                    help="home joint pose to park at on startup [rad] (default: Franka ready)")
    ap.add_argument("--reset-lift", type=float, default=0.15,
                    help="on reset, retract the EE straight UP by this many metres to clear the scene BEFORE homing, "
                         "so the arm doesn't sweep a direct joint path through the bin and topple the reoriented object "
                         "(0 = home directly)")
    ap.add_argument("--no-home", action="store_true", help="don't move to home on startup")
    ap.add_argument("--no-gripper", action="store_true", help="don't close the gripper on startup")
    ap.add_argument("--grip-force", type=float, default=40.0, help="gripper close force [N]")
    ap.add_argument("--home-gripper", action="store_true", help="calibrate the gripper (homing) first")
    ap.add_argument("--yes", action="store_true", help="skip the e-stop safety prompt before homing")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    robot.automatic_error_recovery()               # clear any reflex left from a previous crash so we can start

    # --- park at HOME + close the gripper BEFORE going live with impedance ---
    if not args.no_home:
        if not args.yes:
            input("Workspace clear, e-stop in hand? Enter to move HOME then arm impedance... ")
        print("moving to home...")
        reached, _ = go_home(robot, args.home)
        print(f"  home reached={reached}")
    if not args.no_gripper:
        print("closing gripper...")
        gs = set_gripper(args.ip, "close", force=args.grip_force, do_homing=args.home_gripper)
        print(f"  gripper width={gs.width:.4f} m  is_grasped={gs.is_grasped}")

    def wait_until_still(timeout=2.0, thresh=5e-3, consec=10):
        """Block until the arm is motionless (all |dq| < thresh for `consec` reads in a row).
        The firmware's hold torque must be ~gravity-only when torque control takes over —
        handing off mid-settle is a torque step and trips controller_torque_discontinuity."""
        still = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            still = still + 1 if max(abs(v) for v in robot.read_once().dq) < thresh else 0
            if still >= consec:
                return True
        print(f"  warning: arm not settled after {timeout}s, arming impedance anyway")
        return False

    def make_impedance():
        robot.automatic_error_recovery()           # clear any prior reflex/error so (re)starting control works
        wait_until_still()
        c = JointImpedanceController(robot, max_delta_tau=args.max_delta_tau,
                                     reference_mode=args.reference_mode, max_dq=args.max_dq,
                                     interp_time=args.interp_time)
        c.start()
        return c

    grip = GripperService(args.ip)                 # spawn gripper poll+command processes BEFORE going RT
    if not grip.wait_ready():                      # MUST be connected before the RT session starts (see wait_ready)
        print("  warning: gripper poll not ready after 5 s (gripper off?) — arming anyway")
    sys.setswitchinterval(0.0005)                  # default GIL slice is 5 ms — zmq/json must not hold 5 torque cycles
    gc.freeze(); gc.disable()                      # a gen-2 GC pause is multiple ms; the 1 kHz loop can't afford one
    ctrl = make_impedance()                        # 1 kHz loop begins (now holding at HOME)
    print("joint-impedance controller running (holding at home); set_action enabled")

    def handle(req):
        nonlocal ctrl
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True}
        if cmd == "get_state":
            s = ctrl.get_state()
            rep = {"ok": True, "q": s["q"].tolist(), "dq": s["dq"].tolist(),
                   "ee_pose": s["T_base_ee"].tolist(), "jacobian": s["jacobian"].tolist(),
                   "gripper": grip.mean_finger(),
                   "controlling": s["controlling"]}
            if not s["controlling"] and ctrl._error:   # control loop died
                rep["ok"] = False
                rep["error"] = f"control error: {ctrl._error}"
            return rep
        if cmd == "set_action":
            ctrl.set_action(req["a"])              # non-blocking; 1 kHz loop tracks it
            return {"ok": True}
        if cmd == "gripper":                       # open/close, NON-BLOCKING (spawned
            # process, own GIL — see controllers/gripper_service.py). The torque
            # loop keeps holding/tracking while the fingers move; the reply
            # returns immediately so state polling never stalls mid-grasp.
            started = grip.command(req.get("action", "close"),
                                   width=req.get("width", 0.0),
                                   force=req.get("force", args.grip_force))
            gs = grip.state()
            return {"ok": True, "started": started,
                    "width": gs["width"], "is_grasped": gs["is_grasped"]}
        if cmd == "gripper_sync":                  # RELIABLE gripper, BLOCKING, WITHOUT homing (stop -> set_gripper ->
            # re-arm AT THE CURRENT POSE). Needed because the non-blocking GripperService path is unreliable here: a
            # reset's fresh set_gripper connection steals the single gripper connection, so the worker's lifelong one
            # goes dead and every later `gripper` service command silently fails. set_gripper (own connection, loop
            # stopped -> no comm reflex) always works; skipping go_home lets it also grip with the arm parked at the
            # object (where `reset` would home away). make_impedance re-reads the CURRENT pose -> holds there, no jump.
            ctrl.stop()                            # end the torque loop; firmware idle-holds the arm in place
            action = req.get("action", "close")
            gs = set_gripper(args.ip, action, width=req.get("width", 0.0), force=req.get("force", args.grip_force))
            ctrl = make_impedance()                # recovers + waits-still + re-arms at the current pose (no home)
            return {"ok": True, "width": float(gs.width), "is_grasped": bool(gs.is_grasped)}
        if cmd == "reset":                         # episode reset (BLOCKING): recover -> impedance-home -> gripper -> re-arm
            ctrl.stop()                            # end the current torque loop
            ctrl = make_impedance()                # recovers the reflex + re-arms torque at the CURRENT (loaded) pose:
            #                                        compliant + loose collision -> holds without re-tripping the Reflex.
            _impedance_lift(ctrl, args.reset_lift) # retract straight UP first to CLEAR the scene, so homing doesn't
            #                                        sweep the arm through the bin and topple the reoriented object.
            _impedance_goto(ctrl, args.home)       # THEN ramp to home THROUGH the impedance controller (NOT position
            #                                        control, so a timeout parked on the object can't Reflex-reject it).
            grip_action = req.get("gripper", "close")   # NOT `grip` — that's the GripperService
            if grip_action in ("close", "shut", "open"):
                ctrl.stop()                        # set_gripper opens a fresh Gripper connection -> must NOT be mid-RT session
                set_gripper(args.ip, grip_action, force=args.grip_force)
                ctrl = make_impedance()            # re-arm at home
            return {"ok": True, "mode": "reset-home"}
        return {"ok": False, "error": f"unknown cmd: {cmd!r}"}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(args.bind)
    print(f"impedance_server listening on {args.bind}  (Ctrl-C to stop)")
    try:
        while True:
            try:
                req = sock.recv_json()
            except zmq.Again:
                if ctrl._error:
                    print(f"FATAL: {ctrl._error}")
                    break
                continue
            try:
                rep = handle(req)
            except Exception as e:
                rep = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            sock.send_json(rep)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        ctrl.stop()
        grip.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
