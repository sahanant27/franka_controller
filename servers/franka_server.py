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

Run on the ROBOT PC (e-stop in hand — the arm becomes live on startup):
  python servers/franka_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
"""
import argparse
import os
import sys
import traceback

import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.joint_position_controller import JointPositionController, TargetStreamer


def handle(req, streamer):
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "get_state":
        s = streamer.get_state()
        rep = {"ok": True, "q": s["q"].tolist(), "dq": s["dq"].tolist(),
               "ee_pose": s["T_base_ee"].tolist(),
               "controlling": s["controlling"]}   # False in programming/guiding mode
        if "error" in s:                          # the state-READ loop died (state is stale)
            rep["ok"] = False
            rep["error"] = f"streamer error: {s['error']}"
        return rep
    if cmd == "set_target":
        streamer.update_target(req["q"])      # clamps + bounds happen in the feeder
        return {"ok": True}
    return {"ok": False, "error": f"unknown cmd: {cmd!r}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-vel", type=float, default=0.4, help="max joint velocity [rad/s]")
    ap.add_argument("--tol", type=float, default=0.05, help="goal tolerance [rad]")
    ap.add_argument("--rate", type=float, default=50.0, help="feeder re-send rate [Hz]")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    ctrl = JointPositionController(robot, max_velocity=args.max_vel, goal_tolerance=args.tol)
    streamer = TargetStreamer(ctrl, rate_hz=args.rate)
    streamer.start()                          # arm now actively held at current pose
    print(f"streamer running @ {args.rate} Hz (holding current pose)")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)        # so Ctrl-C is responsive
    sock.bind(args.bind)
    print(f"franka_server listening on {args.bind}  (Ctrl-C to stop)")

    try:
        while True:
            try:
                req = sock.recv_json()
            except zmq.Again:
                if streamer._error:           # surface a dead control thread early
                    print(f"FATAL: {streamer._error}")
                    break
                continue
            try:
                rep = handle(req, streamer)
            except Exception as e:            # per-request boundary: always reply
                rep = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            sock.send_json(rep)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        streamer.stop()                       # stop feeder + release async control
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
