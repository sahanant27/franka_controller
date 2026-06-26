#!/usr/bin/env python3
"""Standalone Franka gripper test — open / close / grasp / home. ROBOT PC (pylibfranka).

The gripper is a SEPARATE connection from the arm, so this needs NO impedance_server / torque
loop running. Use it to isolate gripper behaviour and find working grasp params (width / force /
epsilon) before wiring them into the grasp pipeline.

  python tools/test_gripper.py --ip 172.16.0.2                       # cycle: home -> open -> grasp -> open
  python tools/test_gripper.py --ip 172.16.0.2 --action open
  python tools/test_gripper.py --ip 172.16.0.2 --action grasp --width 0.025 --force 40
  python tools/test_gripper.py --ip 172.16.0.2 --action grasp --width 0 --eps 0    # reproduce the failing call

Put an object between the fingers before a grasp. `is_grasped` tells you if it actually gripped.
NOTE: run this while the arm server is NOT doing gripper ops (ideally idle/stopped) to avoid two
gripper connections at once.
"""
import argparse
import time

import pylibfranka as franka

MAX_WIDTH = 0.08


def show(g, tag):
    s = g.read_once()
    mw = getattr(s, "max_width", float("nan"))
    print(f"  [{tag:8}] width={s.width:.4f} m  is_grasped={s.is_grasped}  max_width={mw:.4f}  "
          f"temp={getattr(s, 'temperature', '?')}")
    return s


def grasp(g, width, speed, force, eps):
    """grasp() with epsilon if the binding exposes it, else the 3-arg form. Returns the bool result."""
    width = min(MAX_WIDTH, max(0.0, width))
    try:
        return g.grasp(width, speed, force, eps, eps)      # libfranka: grasp(width, speed, force, eps_in, eps_out)
    except TypeError:
        print("  (binding has no epsilon args — using 3-arg grasp)")
        return g.grasp(width, speed, force)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--action", default="cycle", choices=["cycle", "home", "open", "close", "grasp"])
    ap.add_argument("--width", type=float, default=0.02, help="grasp target width [m] (close/grasp)")
    ap.add_argument("--speed", type=float, default=0.1, help="finger speed [m/s]")
    ap.add_argument("--force", type=float, default=40.0, help="grasp force [N]")
    ap.add_argument("--eps", type=float, default=0.04, help="grasp tolerance (epsilon_inner=epsilon_outer) [m]")
    ap.add_argument("--no-home", action="store_true", help="skip homing")
    args = ap.parse_args()

    g = franka.Gripper(args.ip)
    print(f"connected to gripper @ {args.ip}")
    show(g, "start")

    if args.action in ("cycle", "home") and not args.no_home:
        print("homing (calibrates max width)...")
        g.homing()
        show(g, "homed")
    if args.action == "home":
        return 0

    if args.action == "open":
        print(f"opening to {MAX_WIDTH} m ...")
        g.move(MAX_WIDTH, args.speed)
        show(g, "open")
        return 0

    if args.action in ("close", "grasp"):
        print(f"grasping: width={args.width} speed={args.speed} force={args.force} eps={args.eps} ...")
        ok = grasp(g, args.width, args.speed, args.force, args.eps)
        print(f"  grasp() returned: {ok}")
        show(g, "grasped")
        return 0

    # cycle: open -> grasp -> open
    print(f"opening to {MAX_WIDTH} m ...")
    g.move(MAX_WIDTH, args.speed); show(g, "open"); time.sleep(0.5)
    print(f"grasping at width={args.width} (put an object between the fingers) ...")
    ok = grasp(g, args.width, args.speed, args.force, args.eps)
    print(f"  grasp() returned: {ok}"); show(g, "grasped"); time.sleep(1.5)
    print("opening again ...")
    g.move(MAX_WIDTH, args.speed); show(g, "open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
