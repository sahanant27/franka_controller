#!/usr/bin/env python3
"""ZeroMQ robot server — runs on the ROBOT PC (the machine with pylibfranka).

REQ/REP: the perception PC sends a JSON command, this replies with JSON.
Commands:
  {"cmd": "ping"}       -> {"ok": true, "pong": true}
  {"cmd": "get_state"}  -> {"ok": true, "O_T_EE": [16 col-major], "q": [7]}
Add more in handle() as you build the grasp loop (e.g. "move_to").

Setup on the robot PC (in the pylibfranka env):
    pip install pyzmq
    python robot_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
Test the Ethernet link WITHOUT the robot (run on either PC):
    python robot_server.py --mock
"""
import argparse
import traceback

import zmq


def make_get_state(args):
    """Return a callable -> {"O_T_EE": [...], "q": [...]}. Mock avoids pylibfranka."""
    if args.mock:
        import numpy as np
        identity = list(np.eye(4).flatten(order="F"))   # column-major, like libfranka
        return lambda: {"O_T_EE": identity, "q": [0.0] * 7}

    from pylibfranka import Robot
    robot = Robot(args.ip)
    s = robot.read_once()
    print(f"connected to robot {args.ip}; q = {[round(v, 3) for v in s.q]}")

    def get_state():
        s = robot.read_once()                # non-realtime read, does not move the arm
        return {"O_T_EE": list(s.O_T_EE), "q": list(s.q)}
    return get_state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2", help="Franka FCI IP")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556",
                    help="0.0.0.0 listens on the Ethernet link, not just localhost")
    ap.add_argument("--mock", action="store_true",
                    help="serve a fake identity pose (test the link without the robot)")
    args = ap.parse_args()

    get_state = make_get_state(args)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)       # so Ctrl-C is responsive
    sock.bind(args.bind)
    print(f"robot_server {'(MOCK) ' if args.mock else ''}listening on {args.bind}  (Ctrl-C to stop)")

    def handle(req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "pong": True}
        if cmd == "get_state":
            return {"ok": True, **get_state()}
        return {"ok": False, "error": f"unknown cmd: {cmd}"}

    try:
        while True:
            try:
                req = sock.recv_json()       # REP must reply once per recv (lockstep)
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
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
