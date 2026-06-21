#!/usr/bin/env python3
"""Policy server for the joint-impedance (manipulation) controller — ROBOT PC.

Same pattern + zmq port as franka_server / calib_server (mutually exclusive: run
whichever the current mode needs). Runs JointImpedanceController (1 kHz torque
loop) and streams 21-D actions from the perception PC's policy.

Protocol (zmq REQ/REP):
  {"cmd":"ping"}                 -> {"ok":true}
  {"cmd":"get_state"}            -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4],"jacobian":[[7]x6],"controlling":bool}
  {"cmd":"set_action","a":[21]}  -> {"ok":true}      (a = [Δq(7), Kp(7), Kd(7)])
  {"cmd":"reset","gripper":"close"|"open"|"none"} -> {"ok":true}  (episode reset: stop -> home -> re-arm)

On startup it parks at HOME and closes the gripper, then holds at home under impedance
(--no-home / --no-gripper to skip; --home-gripper to calibrate the gripper first).
Run on the ROBOT PC, e-stop in hand — the arm becomes live on startup:
  python servers/impedance_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556 --max-dq 0.5
"""
import argparse
import os
import sys
import traceback

import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.joint_impedance_controller import JointImpedanceController
from controllers.home import Q_HOME, go_home, set_gripper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-delta-tau", type=float, default=1.0, help="per-tick torque slew limit [Nm]")
    ap.add_argument("--max-dq", type=float, default=0.5, help="safety clamp on |Δq| per action [rad]")
    ap.add_argument("--interp-time", type=float, default=0.2,
                    help="linear target-interpolation ramp time [s] ~ the policy period (5 Hz -> 0.2). "
                         "Ramps q_ref to the new target so the 1 kHz loop tracks a smooth ramp, not a step.")
    ap.add_argument("--reference-mode", choices=["measured", "commanded"], default="measured")
    ap.add_argument("--home", type=float, nargs=7, default=Q_HOME, metavar="Q",
                    help="home joint pose to park at on startup [rad] (default: Franka ready)")
    ap.add_argument("--no-home", action="store_true", help="don't move to home on startup")
    ap.add_argument("--no-gripper", action="store_true", help="don't close the gripper on startup")
    ap.add_argument("--grip-force", type=float, default=40.0, help="gripper close force [N]")
    ap.add_argument("--home-gripper", action="store_true", help="calibrate the gripper (homing) first")
    ap.add_argument("--yes", action="store_true", help="skip the e-stop safety prompt before homing")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)

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

    def make_impedance():
        c = JointImpedanceController(robot, max_delta_tau=args.max_delta_tau,
                                     reference_mode=args.reference_mode, max_dq=args.max_dq,
                                     interp_time=args.interp_time)
        c.start()
        return c

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
                   "controlling": s["controlling"]}
            if not s["controlling"] and ctrl._error:   # control loop died
                rep["ok"] = False
                rep["error"] = f"control error: {ctrl._error}"
            return rep
        if cmd == "set_action":
            ctrl.set_action(req["a"])              # non-blocking; 1 kHz loop tracks it
            return {"ok": True}
        if cmd == "reset":                         # episode reset (BLOCKING): stop -> home (+gripper) -> re-arm
            ctrl.stop()                            # end the torque loop; firmware idle-holds during the move
            go_home(robot, args.home)              # async position -> home, then released
            grip = req.get("gripper", "close")
            if grip in ("close", "open"):
                set_gripper(args.ip, grip, force=args.grip_force)
            ctrl = make_impedance()                # FRESH controller reads home -> holds there, no jump
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
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
