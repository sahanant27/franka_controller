#!/usr/bin/env python3
"""Calibration server (ROBOT PC) — calibration ONLY, separate from the policy
server (franka_server.py). Same zmq port, so perception-side tools are unchanged.

Two modes:
  --mode teach   Read-only: serves get_state via read_once, starts NO controller,
                 so you can free-drive (guiding mode). For INITIAL hand-eye
                 calibration (capture.py reads pose + captures as you free-drive).
  --mode replay  Starts the async controller (holds the arm) and adds move_to, so
                 the perception PC can drive the arm to stored poses and re-capture.

Protocol (zmq REQ/REP):
  {"cmd":"ping"}              -> {"ok":true}
  {"cmd":"get_state"}         -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4],"controlling":bool}
  {"cmd":"move_to","q":[7]}   -> {"ok":true,"reached":bool,"q":[7]}   (replay only)

  python calib_server.py --mode teach  --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
  python calib_server.py --mode replay --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
"""
import argparse
import time
import traceback

import numpy as np
import zmq
import pylibfranka as franka

from joint_position_controller import JointPositionController, TargetStreamer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["teach", "replay"], required=True)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-vel", type=float, default=0.4)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--rate", type=float, default=50.0)
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    streamer = None

    if args.mode == "replay":
        ctrl = JointPositionController(robot, max_velocity=args.max_vel, goal_tolerance=args.tol)
        streamer = TargetStreamer(ctrl, rate_hz=args.rate)
        streamer.start()
        print("REPLAY mode: controller active (arm held); move_to enabled.")

        def get_state():
            s = streamer.get_state()
            return {"q": s["q"].tolist(), "dq": s["dq"].tolist(),
                    "ee_pose": s["T_base_ee"].tolist(), "controlling": s["controlling"]}

        def move_to(q):
            q = np.asarray(q, dtype=float)
            timeout = float(np.max(np.abs(q - streamer.get_state()["q"])) / args.max_vel) + 3.0
            streamer.update_target(q)
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                cur = streamer.get_state()["q"]
                if np.max(np.abs(cur - q)) <= args.tol:
                    return True, cur.tolist()
                time.sleep(0.02)
            return False, streamer.get_state()["q"].tolist()
    else:
        print("TEACH mode: read-only. Free-drive the arm (guiding mode).")

        def get_state():
            s = robot.read_once()                                 # encoders only, no control
            T = np.array(s.O_T_EE).reshape(4, 4, order="F")
            return {"q": list(s.q), "dq": list(s.dq),
                    "ee_pose": T.tolist(), "controlling": False}

        def move_to(q):
            raise RuntimeError("move_to is only available in --mode replay")

    def handle(req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True}
        if cmd == "get_state":
            return {"ok": True, **get_state()}
        if cmd == "move_to":
            reached, q = move_to(req["q"])
            return {"ok": True, "reached": reached, "q": q}
        return {"ok": False, "error": f"unknown cmd: {cmd!r}"}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(args.bind)
    print(f"calib_server ({args.mode}) listening on {args.bind}  (Ctrl-C to stop)")
    try:
        while True:
            try:
                req = sock.recv_json()
            except zmq.Again:
                if streamer is not None and streamer._error:
                    print(f"FATAL: {streamer._error}")
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
        if streamer is not None:
            streamer.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
