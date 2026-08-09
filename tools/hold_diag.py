#!/usr/bin/env python3
"""Instrumented 1 kHz impedance HOLD — find out exactly who blows the 1 ms deadline.

Runs the same torque-PD hold as JointImpedanceController but in the MAIN thread of a
process with NO other threads, no zmq, no gripper — the minimal reproduction. Every tick
records the duration of readOnce / compute / writeOnce and the total period. On a reflex
it prints when it died and the worst ticks with their breakdown; on a clean run it prints
the timing budget. Bisect ladder: this clean -> add server layers back one at a time.

Run on the ROBOT PC, e-stop in hand (arm goes live, holds current pose):
  python tools/hold_diag.py --secs 120
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.joint_impedance_controller import DEFAULT_KP, DEFAULT_KD, MAX_TORQUES, _LO_TAU, _LO_F


def report(tag, n, rd, cp, wr, pd):
    def pct(a, p):
        s = np.sort(a[:n]) * 1e3
        return s[min(n - 1, int(p / 100 * n))]
    print(f"  [{tag}] {n} ticks:")
    for name, a in (("readOnce", rd), ("compute", cp), ("writeOnce", wr), ("period", pd)):
        print(f"    {name:9s} p50 {pct(a,50):6.3f}  p99 {pct(a,99):6.3f}  p99.9 {pct(a,99.9):6.3f}  max {np.max(a[:n])*1e3:6.3f} ms")
    worst = np.argsort(pd[:n])[-10:][::-1]
    print("    worst periods (tick@t: period = read + compute + write):")
    for i in worst:
        print(f"      #{i}@{i/1000:6.1f}s: {pd[i]*1e3:6.2f} = {rd[i]*1e3:5.2f} + {cp[i]*1e3:5.2f} + {wr[i]*1e3:5.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--secs", type=float, default=120.0)
    ap.add_argument("--fifo", type=int, default=80, help="SCHED_FIFO priority for this thread (0 = don't elevate)")
    ap.add_argument("--jac-every", type=int, default=20,
                    help="compute the jacobian every N ticks like the real loop (0 = never)")
    ap.add_argument("--slew", type=float, default=0.8, help="per-tick torque slew limit [Nm]")
    ap.add_argument("--yes", action="store_true", help="skip the e-stop prompt")
    args = ap.parse_args()

    print(f"kernel={os.uname().release}  fifo={args.fifo}  jac_every={args.jac_every}")
    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    robot.automatic_error_recovery()
    robot.set_collision_behavior(_LO_TAU, _LO_TAU, _LO_F, _LO_F)
    if not args.yes:
        input("Workspace clear, e-stop in hand? Enter to arm the instrumented hold... ")

    if args.fifo:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(args.fifo))
            print(f"SCHED_FIFO {args.fifo} active")
        except OSError as e:
            print(f"SCHED_FIFO denied ({e}) — continuing best-effort")
    sys.setswitchinterval(0.0005)
    gc.freeze(); gc.disable()

    model = robot.load_model()
    q0 = np.array(robot.read_once().q)
    kp, kd = DEFAULT_KP, DEFAULT_KD
    print(f"holding q = {q0.round(3).tolist()}")

    n_max = int(args.secs * 1000)
    rd = np.zeros(n_max); cp = np.zeros(n_max); wr = np.zeros(n_max); pd = np.zeros(n_max)
    prev_tau = np.zeros(7)
    n = 0
    t_last = None
    t_start = time.monotonic()
    active = robot.start_torque_control()
    try:
        while n < n_max:
            t_a = time.perf_counter()
            state, _ = active.readOnce()
            t_b = time.perf_counter()

            q = np.array(state.q)
            dq = np.array(state.dq)
            if args.jac_every and n % args.jac_every == 0:
                _ = np.array(model.zero_jacobian(state)).reshape(6, 7, order="F")
            tau = kp * (q0 - q) - kd * dq
            tau = prev_tau + np.clip(tau - prev_tau, -args.slew, args.slew)
            tau = np.clip(tau, -MAX_TORQUES, MAX_TORQUES)
            prev_tau = tau
            cmd = franka.Torques(tau.tolist())
            cmd.motion_finished = False
            t_c = time.perf_counter()
            active.writeOnce(cmd)
            t_d = time.perf_counter()

            rd[n] = t_b - t_a; cp[n] = t_c - t_b; wr[n] = t_d - t_c
            pd[n] = (t_d - t_last) if t_last is not None else 0.001
            t_last = t_d
            n += 1
            if n % 10000 == 0:
                report(f"t={n/1000:.0f}s", n, rd, cp, wr, pd)
    except Exception as e:
        print(f"\n*** DIED after {time.monotonic()-t_start:.2f}s ({n} ticks): {e}")
        if n > 10:
            report("at death", n, rd, cp, wr, pd)
        return 1
    finally:
        try:
            cmd = franka.Torques(prev_tau.tolist())
            cmd.motion_finished = True
            active.writeOnce(cmd)
        except Exception:
            pass
    print(f"\nCLEAN RUN ({args.secs:.0f}s):")
    report("final", n, rd, cp, wr, pd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
