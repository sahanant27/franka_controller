#!/usr/bin/env python3
"""Validate TargetStreamer + zmq end-to-end (CLIENT for franka_server.py).

The key thing this proves: even though WE send set_target at only --hz (default
10 Hz, your policy rate), the arm moves smoothly because the SERVER's feeder
re-sends the latest target at 50 Hz. If the arm tracks the target here, the
50-Hz-feeder design is confirmed on hardware.

Run franka_server.py first, then this. Start on the ROBOT PC (localhost) to
de-risk, then re-run from the PERCEPTION PC with --addr tcp://10.42.0.1:5556.

  python tools/validate_streamer.py --addr tcp://127.0.0.1:5556 --joint 6 --delta 0.2 --hz 10
"""
import argparse
import time

import numpy as np
import zmq


def request(sock, msg, timeout_ms=2000):
    sock.send_json(msg)
    if sock.poll(timeout_ms) == 0:
        raise TimeoutError(f"no reply to {msg.get('cmd')!r}")
    rep = sock.recv_json()
    if not rep.get("ok"):
        raise RuntimeError(rep.get("error", "server error"))
    return rep


def stream_to(sock, target, hz, duration, joint):
    """Send set_target at hz for `duration` s, polling get_state; return last q."""
    dt = 1.0 / hz
    t0 = time.monotonic()
    q = np.array(target)
    while time.monotonic() - t0 < duration:
        loop = time.monotonic()
        request(sock, {"cmd": "set_target", "q": target.tolist()})
        q = np.array(request(sock, {"cmd": "get_state"})["q"])
        print(f"  t={time.monotonic()-t0:4.1f}s  err={np.max(np.abs(q-target)):.3f} rad  "
              f"q[{joint}]={q[joint]:+.3f}")
        sleep = dt - (time.monotonic() - loop)
        if sleep > 0:
            time.sleep(sleep)
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--addr", default="tcp://127.0.0.1:5556")
    ap.add_argument("--joint", type=int, default=6)
    ap.add_argument("--delta", type=float, default=0.2, help="move this far [rad], then back")
    ap.add_argument("--hz", type=float, default=10.0, help="OUR command rate (policy rate)")
    ap.add_argument("--duration", type=float, default=4.0, help="seconds per leg")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.addr)

    print("ping ->", request(sock, {"cmd": "ping"}))
    q0 = np.array(request(sock, {"cmd": "get_state"})["q"])
    print(f"start q = {q0.round(3).tolist()}")

    target = q0.copy()
    target[args.joint] += args.delta
    print(f"\nstreaming to target (joint {args.joint} += {args.delta}) at {args.hz} Hz...")
    qf = stream_to(sock, target, args.hz, args.duration, args.joint)
    print(f"reached err = {np.max(np.abs(qf - target)):.3f} rad")

    print(f"\nstreaming back to start at {args.hz} Hz...")
    qb = stream_to(sock, q0, args.hz, args.duration, args.joint)
    print(f"return  err = {np.max(np.abs(qb - q0)):.3f} rad")
    print("\nIf the arm tracked smoothly at 10 Hz, the 50 Hz feeder design is confirmed.")
    sock.close(0)
    ctx.term()
