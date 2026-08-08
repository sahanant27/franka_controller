#!/usr/bin/env python3
"""Single server for the full task — owns the FCI and switches control modes (ROBOT PC).

Replaces running franka_server / calib_server / impedance_server one-at-a-time: this holds one
FCI connection and a ModeManager that atomically switches between POSITION (grasp) and IMPEDANCE
(manipulation) control. The perception PC drives the whole grasp→manipulate sequence over one zmq
socket by calling set_mode + the per-mode commands.

Protocol (zmq REQ/REP, JSON):
  {"cmd":"ping"}                          -> {"ok":true}
  {"cmd":"set_mode","mode":"position|impedance|idle"} -> {"ok":true,"mode":...}
  {"cmd":"get_state"}                     -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4],"mode":...,"controlling":bool}
  {"cmd":"set_target","q":[7]}            -> {"ok":true}            (POSITION mode)
  {"cmd":"move_to","q":[7]}               -> {"ok":true,"reached":bool,"q":[7]}  (POSITION mode)
  {"cmd":"set_action","a":[21]}           -> {"ok":true}            (IMPEDANCE mode)

Run on the ROBOT PC (pylibfranka env, e-stop in hand):
  python servers/mode_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556

STRUCTURE: the zmq loop + dispatch are complete; they route to ModeManager, whose lifecycle
bodies are TODO (controllers/mode_manager.py).
"""
import argparse
import os
import sys
import traceback

import numpy as np
import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.mode_manager import ModeManager


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--reference-mode", choices=["measured", "commanded"], default="measured")
    ap.add_argument("--start-mode", choices=list(ModeManager.MODES), default="idle")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    mgr = ModeManager(robot, reference_mode=args.reference_mode)
    if args.start_mode != ModeManager.IDLE:
        mgr.set_mode(args.start_mode)
    print(f"mode_server up (mode={mgr.mode}); switch with set_mode.")

    def handle(req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True}
        if cmd == "set_mode":
            return {"ok": True, "mode": mgr.set_mode(req["mode"])}
        if cmd == "get_state":
            s = mgr.get_state()
            ee = s["T_base_ee"]
            return {"ok": True,
                    "q": np.asarray(s["q"]).tolist(), "dq": np.asarray(s["dq"]).tolist(),
                    "ee_pose": np.asarray(ee).tolist(),
                    "mode": s.get("mode", mgr.mode), "controlling": s.get("controlling")}
        if cmd == "set_target":
            mgr.set_target(req["q"])
            return {"ok": True}
        if cmd == "move_to":
            reached, q = mgr.move_to(req["q"])
            return {"ok": True, "reached": reached, "q": np.asarray(q).tolist()}
        if cmd == "set_action":
            mgr.set_action(req["a"])
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd: {cmd!r}"}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(args.bind)
    print(f"mode_server listening on {args.bind}  (Ctrl-C to stop)")
    try:
        while True:
            try:
                req = sock.recv_json()
            except zmq.Again:
                continue
            try:
                rep = handle(req)
            except Exception as e:
                rep = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            sock.send_json(rep)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        mgr.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    raise SystemExit(main())
