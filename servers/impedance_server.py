#!/usr/bin/env python3
"""Policy server for the joint-impedance (manipulation) controller — ROBOT PC.

Same pattern + zmq port as franka_server / calib_server (mutually exclusive: run
whichever the current mode needs). Runs JointImpedanceController (1 kHz torque
loop) and streams 21-D actions from the perception PC's policy.

Protocol (zmq REQ/REP):
  {"cmd":"ping"}                 -> {"ok":true}
  {"cmd":"get_state"}            -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4],"controlling":bool}
  {"cmd":"set_action","a":[21]}  -> {"ok":true}      (a = [Δq(7), Kp(7), Kd(7)])

Run on the ROBOT PC (e-stop in hand — the arm becomes live on startup):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-delta-tau", type=float, default=1.0, help="per-tick torque slew limit [Nm]")
    ap.add_argument("--max-dq", type=float, default=0.5, help="safety clamp on |Δq| per action [rad]")
    ap.add_argument("--reference-mode", choices=["measured", "commanded"], default="measured")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    ctrl = JointImpedanceController(robot, max_delta_tau=args.max_delta_tau,
                                    reference_mode=args.reference_mode, max_dq=args.max_dq)
    ctrl.start()                                   # 1 kHz loop begins (holds current pose)
    print("joint-impedance controller running (holding current pose); set_action enabled")

    def handle(req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True}
        if cmd == "get_state":
            s = ctrl.get_state()
            rep = {"ok": True, "q": s["q"].tolist(), "dq": s["dq"].tolist(),
                   "ee_pose": s["T_base_ee"].tolist(), "controlling": s["controlling"]}
            if not s["controlling"] and ctrl._error:   # control loop died
                rep["ok"] = False
                rep["error"] = f"control error: {ctrl._error}"
            return rep
        if cmd == "set_action":
            ctrl.set_action(req["a"])              # non-blocking; 1 kHz loop tracks it
            return {"ok": True}
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
