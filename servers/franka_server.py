#!/usr/bin/env python3
"""Fresh RT-server — async joint-position control over zmq (ROBOT PC).

Owns the robot, a JointPositionController (async handler), and a TargetStreamer
(50 Hz feeder thread that keeps async control alive — see joint_position_controller.py
for why a single set_target does NOT hold). The zmq REP loop runs in the main
thread and only touches the streamer's thread-safe methods.

Protocol (JSON over zmq REQ/REP):
  {"cmd":"ping"}                -> {"ok":true}
  {"cmd":"get_state"}           -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4]}
  {"cmd":"set_target","q":[7]}  -> {"ok":true}     (updates the streamed target)
  {"cmd":"recover"}             -> {"ok":true}     (reflex/collision recovery: clear the
                                   error, re-arm the controller, hold current pose)
  {"cmd":"gripper","action":"open"|"close"[,"width","force"]}
                                -> {"ok":true,"width":<mean finger pos>}  (returns
                                   IMMEDIATELY; the open/close runs in the background so
                                   the 50 Hz state polling never stalls)

get_state additionally reports "gripper": mean finger position [m] (width/2,
~0.04 = fully open), or null with --no-gripper.

Run on the ROBOT PC (e-stop in hand — the arm becomes live on startup):
  python servers/franka_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
"""
import argparse
import multiprocessing
import os
import sys
import threading
import time
import traceback

import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.gripper_service import GripperService
from controllers.joint_position_controller import JointPositionController, TargetStreamer
from controllers.streamed_joint_position_controller import StreamedJointPositionController


def handle(req, streamer, grip):
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "get_state":
        s = streamer.get_state()
        rep = {"ok": True, "q": s["q"].tolist(), "dq": s["dq"].tolist(),
               "ee_pose": s["T_base_ee"].tolist(),
               "gripper": grip.mean_finger() if grip else None,
               "controlling": s["controlling"]}   # False in programming/guiding mode
        if "error" in s:                          # the state-READ loop died (state is stale)
            rep["ok"] = False
            rep["error"] = f"streamer error: {s['error']}"
        return rep
    if cmd == "set_target":
        streamer.update_target(req["q"])      # clamps + bounds happen in the feeder
        return {"ok": True}
    if cmd == "gripper":
        if grip is None:
            return {"ok": False, "error": "gripper disabled (--no-gripper)"}
        started = grip.command(req.get("action", "open"),
                               width=float(req.get("width", 0.0)),
                               force=float(req.get("force", 40.0)))
        return {"ok": True, "started": started, "width": grip.mean_finger()}
    return {"ok": False, "error": f"unknown cmd: {cmd!r}"}


def arm_streamer(robot, args):
    """Start the selected executor (holds current pose). Both expose the same
    update_target/get_state/stop/_error interface.

    async:    AsyncPositionControlHandler + 50 Hz feeder — validated for SPARSE
              point-to-point targets; it decelerates at every target, so a
              continuous stream judders (start-stop).
    streamed: 1 kHz reference-tracking loop (writeOnce JointPositions) — for
              continuously streamed targets (policies); reference glides with
              velocity continuity, no braking between targets. 1 kHz-fresh state.
    """
    if args.controller == "streamed":
        return StreamedJointPositionController(robot, tau=args.tau,
                                               max_velocity=args.max_vel)
    ctrl = JointPositionController(robot, max_velocity=args.max_vel, goal_tolerance=args.tol)
    streamer = TargetStreamer(ctrl, rate_hz=args.rate)
    streamer.start()
    return streamer


def recover(robot, streamer, args):
    """Reflex/collision/e-stop recovery: release the (dead) controller, clear the
    robot error, re-arm fresh. The new streamer holds wherever the arm now is."""
    try:
        streamer.stop()                       # joins feeder; releases control if any
    except Exception:
        pass                                  # control may already be gone (reflex)
    robot.automatic_error_recovery()
    return arm_streamer(robot, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-vel", type=float, default=0.4, help="max joint velocity [rad/s]")
    ap.add_argument("--tol", type=float, default=0.05, help="goal tolerance [rad]")
    ap.add_argument("--rate", type=float, default=50.0, help="feeder re-send rate [Hz] (async)")
    ap.add_argument("--controller", choices=["async", "streamed"], default="async",
                    help="async (default): compiled point-to-point goal seeker — robust, "
                         "but brakes at each target. streamed: 1 kHz reference tracking "
                         "(no start-stop) — EXPERIMENTAL until the machine passes "
                         "tools/rt_jitter.py (comm-constraints reflex otherwise).")
    ap.add_argument("--tau", type=float, default=0.06,
                    help="streamed: reference tracker time constant [s] "
                         "(bigger = smoother = laggier)")
    ap.add_argument("--no-gripper", action="store_true",
                    help="no Franka Hand attached / skip gripper support")
    args = ap.parse_args()

    if args.controller == "streamed":
        # The 1 kHz loop must answer every 1 ms. Python's DEFAULT GIL switch
        # interval is 5 ms — any other thread (zmq REP, gripper poll) could hold
        # the GIL for five control cycles and trip the firmware's
        # communication_constraints_violation reflex. Sub-ms handoffs fix that.
        sys.setswitchinterval(0.0005)
        import gc
        gc.freeze()                           # move startup objects out of GC's reach
        gc.disable()                          # a gen-2 collection pauses for MILLISECONDS
        try:
            os.nice(-10)                      # favor us over other processes (best effort)
        except (OSError, PermissionError):
            pass

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    streamer = arm_streamer(robot, args)      # arm now actively held at current pose
    grip = None if args.no_gripper else GripperService(args.ip)
    print(f"streamer running @ {args.rate} Hz (holding current pose)"
          + ("" if grip else " — gripper disabled"))

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)        # so Ctrl-C is responsive
    sock.bind(args.bind)
    print(f"franka_server listening on {args.bind}  (Ctrl-C to stop)")

    warned_error = False
    try:
        while True:
            try:
                req = sock.recv_json()
            except zmq.Again:
                if streamer._error and not warned_error:   # dead read loop: stay up —
                    print(f"streamer error (recoverable via 'recover'): {streamer._error}")
                    warned_error = True                    # the client can send recover
                continue
            try:
                if req.get("cmd") == "recover":
                    streamer = recover(robot, streamer, args)
                    warned_error = False
                    print("recovered: error cleared, controller re-armed")
                    rep = {"ok": True}
                else:
                    rep = handle(req, streamer, grip)
            except Exception as e:            # per-request boundary: always reply
                rep = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            sock.send_json(rep)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        streamer.stop()                       # stop feeder + release async control
        if grip is not None:
            grip.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
