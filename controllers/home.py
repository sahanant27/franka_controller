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
        return ctrl.move_to(np.asarray(q_home, dtype=float), ease=True)   # min-jerk: gentle start + soft arrival
    finally:
        ctrl.stop()


def set_gripper(ip, action="close", width=0.0, speed=0.1, force=40.0, do_homing=False):
    """Gripper control on a SEPARATE connection. action: 'close' | 'open' | 'home'. Returns GripperState."""
    g = franka.Gripper(ip)
    if do_homing or action == "home":
        g.homing()
        if action == "home":
            return g.read_once()
    if action == "open":
        g.move(GRIPPER_MAX_WIDTH, speed)
    else:                                            # close: grasp clamps with force (object or shut)
        g.grasp(width, speed, force)
    return g.read_once()
