#!/usr/bin/env python3
"""STEP 1 — minimal async joint-position move (the API that ROS uses, which our
old joint-pose control was missing).

Goal of this step: confirm AsyncPositionControlHandler actually moves the arm,
and observe what get_target_feedback().status reports (we don't know its values
yet — this prints them so Step 2 can detect "reached" correctly).

SAFETY: moves ONE joint a small amount, slowly. Clear the workspace, keep the
e-stop in hand. Run on the ROBOT PC (in the pylibfranka env).

    python step1_async_joint_move.py --ip 172.16.0.2 --joint 6 --delta 0.2
"""
import argparse
import time

import numpy as np
import pylibfranka as franka


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--joint", type=int, default=6, help="which joint to nudge (0-6)")
    ap.add_argument("--delta", type=float, default=0.2, help="how far to move it [rad]")
    ap.add_argument("--max-vel", type=float, default=0.4, help="max joint velocity [rad/s]")
    ap.add_argument("--tol", type=float, default=0.05, help="goal tolerance [rad]")
    ap.add_argument("--hz", type=float, default=50.0,
                    help="command rate (example uses 50; try 10 to match your policy)")
    ap.add_argument("--duration", type=float, default=8.0, help="max time to run [s]")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    # Conservative collision thresholds so unexpected contact triggers a stop.
    robot.set_collision_behavior(
        [20, 20, 18, 18, 16, 14, 12], [20, 20, 18, 18, 16, 14, 12],
        [20, 20, 20, 25, 25, 25],     [20, 20, 20, 25, 25, 25])

    q0 = np.array(robot.read_once().q)
    target = q0.copy()
    target[args.joint] += args.delta
    print(f"start  q = {q0.round(3).tolist()}")
    print(f"target q = {target.round(3).tolist()}  (joint {args.joint} += {args.delta} rad)")
    input("Workspace clear, e-stop in hand? Press Enter to move... ")

    cfg = franka.AsyncPositionControlHandler.Configuration(
        maximum_joint_velocities=[args.max_vel] * 7, goal_tolerance=args.tol)
    result = franka.AsyncPositionControlHandler.configure(robot, cfg)
    if result.error_message:
        print(f"configure FAILED: {result.error_message}")
        return 1
    handler = result.handler
    tgt = franka.AsyncPositionControlHandler.JointPositionTarget(
        joint_positions=target.tolist())

    dt = 1.0 / args.hz   # command period; example uses 50 Hz, your policy may be 10 Hz
    t0 = time.monotonic()
    last_status = "<none>"
    try:
        while time.monotonic() - t0 < args.duration:
            loop = time.monotonic()
            cmd = handler.set_joint_position_target(tgt)
            if cmd.error_message:
                print(f"set_target error: {cmd.error_message}")
                break
            fb = handler.get_target_feedback()
            if repr(fb.status) != last_status:        # print status only when it changes
                last_status = repr(fb.status)
                print(f"  t={time.monotonic()-t0:4.1f}s  status={fb.status!r}  "
                      f"err={fb.error_message!r}")
            sleep = dt - (time.monotonic() - loop)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        handler.stop_control()

    qf = np.array(robot.read_once().q)
    moved = qf[args.joint] - q0[args.joint]
    print(f"final  q = {qf.round(3).tolist()}")
    print(f"joint {args.joint} moved {moved:+.3f} rad (wanted {args.delta:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
