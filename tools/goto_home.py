#!/usr/bin/env python3
"""Go to a home joint pose and close (or open) the gripper — ROBOT PC, pylibfranka.

Moves the arm to a fixed home configuration via the async joint-position controller, then
closes the Franka Hand. Use to park the robot in a known state. (Same helpers the servers
call on startup — see controllers/home.py.)

  python tools/goto_home.py --ip 172.16.0.2                 # home + close gripper
  python tools/goto_home.py --ip 172.16.0.2 --open          # home + open gripper
  python tools/goto_home.py --ip 172.16.0.2 --no-gripper    # move only
  python tools/goto_home.py --ip 172.16.0.2 --home-gripper  # calibrate gripper (homing) first

Home defaults to the Franka "ready" pose; override with --home q1 .. q7.
SAFETY: the move to home can be large — clear the workspace, keep the e-stop in hand.
NOTE: run with NO other server holding the FCI (this opens its own connection).
"""
import argparse
import os
import sys

import numpy as np
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.home import Q_HOME, go_home, set_gripper


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--home", type=float, nargs=7, default=Q_HOME, metavar="Q",
                    help="home joint config [rad] (default: Franka ready pose)")
    ap.add_argument("--max-vel", type=float, default=0.4, help="max joint velocity [rad/s]")
    ap.add_argument("--tol", type=float, default=0.02, help="goal tolerance [rad]")
    ap.add_argument("--no-gripper", action="store_true", help="move only, leave the gripper alone")
    ap.add_argument("--open", action="store_true", help="open the gripper instead of closing")
    ap.add_argument("--home-gripper", action="store_true", help="calibrate the gripper (homing) first")
    ap.add_argument("--width", type=float, default=0.0, help="close-to width [m] (0 = fully close)")
    ap.add_argument("--speed", type=float, default=0.1, help="gripper speed [m/s]")
    ap.add_argument("--force", type=float, default=40.0, help="grasp force [N]")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    q0 = np.array(robot.read_once().q)
    q_home = np.array(args.home, dtype=float)
    print(f"start q = {q0.round(3).tolist()}")
    print(f"home  q = {q_home.round(3).tolist()}  (largest joint move {np.max(np.abs(q_home - q0)):.2f} rad)")
    input("Workspace clear, e-stop in hand? Press Enter to move HOME... ")

    reached, qf = go_home(robot, args.home, max_vel=args.max_vel, tol=args.tol)
    print(f"reached={reached}  q={np.array(qf).round(3).tolist()}")

    if args.no_gripper:
        return 0
    action = "open" if args.open else "close"
    print(f"gripper: {action} ...")
    gs = set_gripper(args.ip, action, width=args.width, speed=args.speed, force=args.force,
                     do_homing=args.home_gripper)
    print(f"  width={gs.width:.4f} m  is_grasped={gs.is_grasped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
