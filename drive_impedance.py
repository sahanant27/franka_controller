#!/usr/bin/env python3
"""Drive JointImpedanceController with a CONTINUOUS 10 Hz action stream (ROBOT PC).

A local stand-in for the policy: instead of a network policy, this generates a
21-D action [Δq(7), Kp(7), Kd(7)] every 1/hz seconds and calls set_action — to
verify continuous streaming impedance control on hardware BEFORE wiring up the
zmq policy server.

It tracks an absolute target by sending Δq = target - q_measured each step:
  hold : target = q_start            -> stiff hold; push by hand to feel the spring-back.
  sine : target = q_start, one joint += amp*sin(2*pi*freq*t)  -> smooth oscillation.

Run on the ROBOT PC, e-stop in hand:
  python drive_impedance.py --ip 172.16.0.2 --mode hold
  python drive_impedance.py --ip 172.16.0.2 --mode sine --joint 6 --amp 0.15 --freq 0.2
"""
import argparse
import math
import time

import numpy as np
import pylibfranka as franka

from joint_impedance_controller import JointImpedanceController, DEFAULT_KP, DEFAULT_KD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--mode", choices=["hold", "sine"], default="hold")
    ap.add_argument("--joint", type=int, default=6, help="joint to oscillate (sine mode)")
    ap.add_argument("--amp", type=float, default=0.15, help="oscillation amplitude [rad]")
    ap.add_argument("--freq", type=float, default=0.2, help="oscillation frequency [Hz]")
    ap.add_argument("--hz", type=float, default=10.0, help="ACTION rate (policy rate)")
    ap.add_argument("--duration", type=float, default=15.0, help="run time [s]")
    ap.add_argument("--stiffness", type=float, default=None,
                    help="uniform Kp for all joints (default: per-joint DEFAULT_KP)")
    ap.add_argument("--max-dq", type=float, default=0.5, help="safety clamp on |Δq| [rad]")
    args = ap.parse_args()

    if args.stiffness is None:
        kp, kd = DEFAULT_KP.copy(), DEFAULT_KD.copy()
    else:
        kp = np.full(7, args.stiffness)
        kd = 2.0 * np.sqrt(kp)

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    ctrl = JointImpedanceController(robot, max_dq=args.max_dq)
    q_start = ctrl.get_state()["q"]
    print(f"start q = {q_start.round(3).tolist()}  mode={args.mode}")
    input("Workspace clear, e-stop in hand? Press Enter to start streaming... ")

    ctrl.start()                                   # 1 kHz torque loop begins (holds q_start)
    dt = 1.0 / args.hz
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.duration:
            loop = time.monotonic()
            t = loop - t0
            q = ctrl.get_state()["q"]

            target = q_start.copy()
            if args.mode == "sine":
                target[args.joint] = q_start[args.joint] + args.amp * math.sin(2 * math.pi * args.freq * t)

            dq = target - q                        # absolute tracking via delta-from-measured
            action = np.concatenate([dq, kp, kd])  # 21-D
            ctrl.set_action(action)                # non-blocking; loop tracks it for the next ~10 ticks

            st = ctrl.get_state()
            if not st["controlling"]:
                print(f"  control stopped: {ctrl._error}")
                break
            print(f"  t={t:4.1f}s  q[{args.joint}]={q[args.joint]:+.3f}  "
                  f"target={target[args.joint]:+.3f}")
            sleep = dt - (time.monotonic() - loop)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        ctrl.stop()
    print(f"final q = {ctrl.get_state()['q'].round(3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
