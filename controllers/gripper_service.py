#!/usr/bin/env python3
"""Non-blocking gripper access shared by the servers.

Reads: a background thread with its OWN gripper connection caches the width
(~10 Hz) so get_state can attach it for free. Commands: executed in a SEPARATE
PROCESS (spawn) via home.set_gripper — NOT a thread: pylibfranka's blocking
grasp/move holds the GIL for ~1 s, which would starve a 1 kHz control loop
(torque OR position) and stall the zmq REP loop (client staleness guards trip
at the first grasp). A spawned process has its own GIL and its own gripper
connection; the zmq reply returns immediately, mirroring the ROS stack's
send_goal_async semantics. The arm's control loop keeps running throughout —
no stop/restart around gripper motion.
"""
import multiprocessing
import threading
import time

import pylibfranka as franka

from controllers.home import set_gripper


class GripperService:
    def __init__(self, ip, poll_s=0.1):
        self.ip = ip
        self._lock = threading.Lock()
        self._width = None
        self._is_grasped = None
        self._proc = None
        self._ctx = multiprocessing.get_context("spawn")   # never fork libfranka state
        self._running = True
        threading.Thread(target=self._poll, args=(poll_s,), daemon=True,
                         name="gripper_poll").start()

    def _poll(self, poll_s):
        g = franka.Gripper(self.ip)
        while self._running:
            try:
                gs = g.read_once()
                with self._lock:
                    self._width = float(gs.width)
                    self._is_grasped = bool(gs.is_grasped)
            except Exception:
                pass                              # transient read hiccup: keep last value
            time.sleep(poll_s)

    def mean_finger(self):
        """Mean finger position [m] = width/2 (matches /joint_states[7]); None until read."""
        with self._lock:
            return None if self._width is None else self._width / 2.0

    def state(self):
        with self._lock:
            return {"width": self._width, "is_grasped": self._is_grasped}

    def command(self, action, width=0.0, force=40.0):
        """Fire-and-forget open/close in a spawned process. One at a time."""
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return False
            self._proc = self._ctx.Process(
                target=set_gripper, daemon=True,
                kwargs=dict(ip=self.ip, action=action, width=float(width),
                            force=float(force)))
            self._proc.start()
        return True

    def stop(self):
        self._running = False
