#!/usr/bin/env python3
"""TEACH hand-eye calibration poses by free-drive (ROBOT PC, read-only).

Run with franka_server.py OFF and the arm in GUIDING mode (hold the guiding
buttons on the wrist to free-drive). Hand-move the arm so the gripper board
faces the anchor camera, press Enter to store that joint config. Repeat for
~15-20 varied poses (vary orientation a lot), then 'd' to save.

This script NEVER controls the robot -- it only calls read_once() -- so it does
not fight guiding mode. Turn franka_server.py back ON afterwards for the
replay + capture step.

  python tools/record_poses.py --ip 172.16.0.2 --out handeye_poses.json
"""
import argparse
import json

import pylibfranka as franka


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--out", default="handeye_poses.json")
    args = ap.parse_args()

    robot = franka.Robot(args.ip)          # no control session -> guiding mode is free
    print("Free-drive the arm (guiding buttons). Position the board toward the "
          "anchor camera, then press Enter to store. 'u'=undo last, 'd'=done.")

    poses = []
    while True:
        cmd = input(f"[{len(poses)} stored]  Enter=store  u=undo  d=done: ").strip().lower()
        if cmd == "d":
            break
        if cmd == "u":
            if poses:
                poses.pop()
                print(f"  removed last; {len(poses)} left")
            continue
        q = [float(v) for v in robot.read_once().q]
        poses.append(q)
        print(f"  stored #{len(poses)}: {[round(v, 3) for v in q]}")

    with open(args.out, "w") as f:
        json.dump({"q": poses}, f, indent=2)
    print(f"\nSaved {len(poses)} poses to {args.out}")
    print("Next: copy this file to the perception PC, then run capture_handeye.py.")


if __name__ == "__main__":
    main()
