#!/usr/bin/env python3
"""Startup helpers (pylibfranka): move the arm to a home joint pose, and close/open the gripper.

Shared by tools/goto_home.py (CLI) and the servers (e.g. impedance_server parks here on startup).
go_home() uses the async joint-position controller and STOPS it afterward, so the caller can then
start a different controller (torque/impedance) on the same robot connection.
"""
import numpy as np
import pylibfranka as franka

from controllers.joint_position_controller import JointPositionController

# Franka "ready" / neutral joint pose [rad].
Q_HOME = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
GRIPPER_MAX_WIDTH = 0.08          # Franka Hand fully open [m]


def go_home(robot, q_home=Q_HOME, max_vel=0.4, tol=0.02):
    """Move the arm to q_home via async joint-position control (blocking). Returns (reached, qf).
    Releases the position controller before returning so another controller can take `robot`."""
    ctrl = JointPositionController(robot, max_velocity=max_vel, goal_tolerance=tol)
    try:
        return ctrl.move_to(np.asarray(q_home, dtype=float), ease=True)   # eased: prompt start + soft arrival
    finally:
        ctrl.stop()


def gripper_do(g, action="close", width=0.0, speed=0.1, force=40.0, do_homing=False):
    """set_gripper's body on an EXISTING connection (GripperService's worker reuses its
    lifelong connection — spawning/connecting mid-RT-session trips the comm reflex).
    action: 'close' | 'shut' | 'open' | 'home'.
    'close' GRASPS (parks at the width target + applies force — for holding an object);
    'shut' fully closes the fingers via a position move (non-prehensile pusher tool, no object).
    Returns GripperState."""
    if do_homing or action == "home":
        g.homing()
        if action == "home":
            return g.read_once()
    if action == "open":
        g.move(GRIPPER_MAX_WIDTH, speed)
    elif action == "shut":                           # fully close the fingers (pusher) — position move, NO grasp/force,
        g.move(0.0, speed)                           # so it goes to 0 (grasp(0.005) instead PARKS at ~5 mm = looks "open")
    else:                                            # close: grasp the object with force
        # grasp(width=0) is rejected by the firmware (and won't move) — grasp at a non-zero target
        # (the object's measured width); the fingers stop on the object and apply `force`.
        w = min(GRIPPER_MAX_WIDTH, max(0.005, float(width)))
        try:                                         # libfranka grasp(width, speed, force, eps_inner, eps_outer):
            g.grasp(w, speed, force, 0.04, 0.04)     # +/-4 cm tolerance -> grips even if the width estimate is off
        except TypeError:                            # binding doesn't expose epsilon args -> 3-arg form
            g.grasp(w, speed, force)
    return g.read_once()


def set_gripper(ip, action="close", width=0.0, speed=0.1, force=40.0, do_homing=False):
    """One-shot gripper op on its OWN connection — for startup/reset paths only
    (no RT session live). While the arm is armed, go through GripperService instead."""
    return gripper_do(franka.Gripper(ip), action, width, speed, force, do_homing)
