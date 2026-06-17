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

  python servers/calib_server.py --mode teach  --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
  python servers/calib_server.py --mode replay --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
"""
import argparse
import os
import sys
import threading
import time
import traceback

import numpy as np
import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.joint_position_controller import JointPositionController, TargetStreamer


class StateReader:
    """Read-only background reader for TEACH / free-drive mode.

    read_once() can hand back a one-call-stale buffer; called sparsely (once per captured pose)
    that became a full one-pose lag in hand-eye (the EE<->board off-by-one). A steady background
    loop keeps the latest snapshot fresh (~ms), so get_state always returns the current pose.
    Reads work in guiding/programming mode and need no controller, so free-drive is unaffected —
    this is the read-only half of TargetStreamer."""

    def __init__(self, robot, rate_hz=100.0):
        self.robot = robot
        self.dt = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._state = None
        self._running = False
        self._thread = None

    def _loop(self):
        while self._running:
            t = time.monotonic()
            try:
                s = self.robot.read_once()
                T = np.array(s.O_T_EE).reshape(4, 4, order="F")
                with self._lock:
                    self._state = {"q": list(s.q), "dq": list(s.dq), "ee_pose": T.tolist()}
            except Exception:
                pass                                  # transient read hiccup; keep the last snapshot
            sleep = self.dt - (time.monotonic() - t)
            if sleep > 0:
                time.sleep(sleep)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="state_reader")
        self._thread.start()
        for _ in range(200):                          # wait up to ~2s for the first snapshot
            with self._lock:
                if self._state is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError("StateReader: no robot state within 2s")

    def get_state(self):
        with self._lock:
            return {**self._state, "controlling": False}

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


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
    reader = None

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
        print("TEACH mode: read-only, continuous reader. Free-drive the arm (guiding mode).")
        reader = StateReader(robot)
        reader.start()
        get_state = reader.get_state                              # always the freshest snapshot

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
        if reader is not None:
            reader.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
