#!/usr/bin/env python3
"""Atomic mode manager — one FCI connection, switch control modes for the full task.

The grasp + nonprehensile-manipulation task needs two control modes on the SAME robot:
  - POSITION  (async joint-position) for scripted moves / pre-grasp / grasp approach.
  - IMPEDANCE (1 kHz variable-impedance torque) for the manipulation policy (21-D action).
libfranka allows only ONE active controller at a time, so a switch must STOP the current
controller and START the next atomically (and hold-in-place across the switch — no jump).
This owns that lifecycle so a SINGLE server can serve the whole task instead of the three
mutually-exclusive servers (franka_server / calib_server / impedance_server).

STRUCTURE ONLY — the lifecycle bodies are TODO (fill in separately). The command routing,
mode constants, and state interface are in place so the server can be wired now.
"""
import threading

import numpy as np

from controllers.joint_position_controller import JointPositionController, TargetStreamer
from controllers.joint_impedance_controller import JointImpedanceController


class ModeManager:
    POSITION = "position"        # async joint-position (set_target / move_to)
    IMPEDANCE = "impedance"      # 1 kHz variable-impedance (set_action [Δθ,Kp,Kd])
    IDLE = "idle"                # no controller; read-only state
    MODES = (POSITION, IMPEDANCE, IDLE)

    def __init__(self, robot, reference_mode="measured"):
        self.robot = robot
        self.reference_mode = reference_mode
        self._lock = threading.Lock()
        self.mode = self.IDLE
        self._pos = None            # (JointPositionController, TargetStreamer) in POSITION mode
        self._imp = None            # JointImpedanceController in IMPEDANCE mode
        # TODO: in IDLE, proprioception must still be FRESH — reuse the StateReader pattern
        #       (continuous read), never a sparse read_once (see the franka-readonce memory).
        self._reader = None

    # ── mode lifecycle ────────────────────────────────────────────────────────────────────────
    def set_mode(self, mode):
        """Atomically switch to `mode`. Returns the new mode.

        TODO:
          - validate mode in MODES; if already in `mode`, no-op.
          - _teardown() the active controller and WAIT for the FCI to release (single-controller).
          - start the requested controller FROM the current measured q (hold in place, no jump).
          - update self.mode under self._lock.
        """
        raise NotImplementedError("ModeManager.set_mode")

    def _start_position(self):
        """TODO: JointPositionController(robot) + TargetStreamer(ctrl); streamer.start().
        Store as self._pos. Target starts at the current q (hold)."""
        raise NotImplementedError

    def _start_impedance(self):
        """TODO: JointImpedanceController(robot, reference_mode=self.reference_mode); .start().
        Store as self._imp. First action latches q_ref = q_meas (hold)."""
        raise NotImplementedError

    def _teardown(self):
        """TODO: stop whichever controller is active (streamer.stop() / imp.stop()), clear handles,
        and leave the arm safely held/idle before the next start."""
        raise NotImplementedError

    # ── commands (routed by mode; error if wrong mode) ────────────────────────────────────────
    def set_target(self, q):
        """POSITION only. TODO: assert self.mode == POSITION; self._pos[1].update_target(q)."""
        raise NotImplementedError

    def move_to(self, q):
        """POSITION only, blocking. Returns (reached, q). TODO."""
        raise NotImplementedError

    def set_action(self, a21):
        """IMPEDANCE only. TODO: assert self.mode == IMPEDANCE; self._imp.set_action(a21)."""
        raise NotImplementedError

    def gripper(self, cmd, **kw):
        """TODO (grasp, item #4): open/close/grasp via franka.Gripper (SEPARATE connection from FCI)."""
        raise NotImplementedError

    # ── state (any mode) ──────────────────────────────────────────────────────────────────────
    def get_state(self):
        """{q, dq, T_base_ee, mode, controlling}. TODO: pull from the active controller's get_state
        (POSITION → streamer, IMPEDANCE → imp); in IDLE use the continuous reader. Always fresh."""
        raise NotImplementedError

    def stop(self):
        """TODO: _teardown() + release the robot / stop the reader."""
        raise NotImplementedError
