#!/usr/bin/env python3
"""Non-blocking gripper access shared by the servers.

NOTHING gripper-related may run in the control process, and after arming it may
never spawn a process or open a gripper connection either:
  - pylibfranka's gripper calls hold the GIL (measured: a single read_once can
    block 80 ms = 80 missed 1 kHz cycles = communication_constraints_violation),
    so in-process threads are out.
  - a fresh gripper TCP connect / interpreter boot disturbs the robot enough to
    trip the same reflex (bisected at session start; reproduced at command time
    — COMM_VIOLATION_DEBUG.md §2/§3), so per-command process spawns are out too.

Hence ONE persistent worker process, started before arming, owning poll AND
commands on a single lifelong gripper connection. Commands go over a queue and
return immediately (mirroring the ROS stack's send_goal_async semantics); state
(width / is_grasped) comes back through shared memory. The arm's control loop
keeps running throughout.
"""
import multiprocessing
import queue
import time

from controllers.home import gripper_do


def _worker(ip, poll_s, width_v, grasped_v, busy_v, running_v, cmd_q):
    """Worker process body (own GIL): one lifelong connection for poll + commands."""
    import pylibfranka as franka
    g = franka.Gripper(ip)
    while running_v.value:
        try:                                  # doubles as the poll pacing
            cmd = cmd_q.get(timeout=poll_s)
        except (queue.Empty, InterruptedError):
            cmd = None
        if cmd is not None:
            busy_v.value = 1
            try:
                gripper_do(g, **cmd)
            except Exception:
                pass                          # failed grasp/move: state below still refreshes
            busy_v.value = 0
        try:
            gs = g.read_once()
            width_v.value = float(gs.width)
            grasped_v.value = int(gs.is_grasped)
        except Exception:
            pass                              # transient read hiccup: keep last value


class GripperService:
    def __init__(self, ip, poll_s=0.1):
        self.ip = ip
        self._ctx = multiprocessing.get_context("spawn")   # never fork libfranka state
        self._width_v = self._ctx.Value("d", -1.0)         # < 0 = no read yet
        self._grasped_v = self._ctx.Value("i", -1)
        self._busy_v = self._ctx.Value("i", 0)
        self._running_v = self._ctx.Value("i", 1)
        self._cmd_q = self._ctx.Queue(maxsize=4)
        self._worker = self._ctx.Process(
            target=_worker, daemon=True,
            args=(ip, poll_s, self._width_v, self._grasped_v,
                  self._busy_v, self._running_v, self._cmd_q))
        self._worker.start()

    def wait_ready(self, timeout=5.0):
        """Block until the worker has its first reading (= gripper connection is up).
        Arming the robot's RT session while the gripper child is still connecting trips
        communication_constraints_violation at session start (bisected 2026-08-08:
        settled service + 1 kHz loop = clean; connecting service + loop start = dead in
        0.5 s)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._width_v.value >= 0:
                return True
            time.sleep(0.05)
        return False

    def mean_finger(self):
        """Mean finger position [m] = width/2 (matches /joint_states[7]); None until read."""
        w = self._width_v.value
        return None if w < 0 else w / 2.0

    def state(self):
        w, g = self._width_v.value, self._grasped_v.value
        return {"width": None if w < 0 else w,
                "is_grasped": None if g < 0 else bool(g)}

    def command(self, action, width=0.0, force=40.0):
        """Queue an open/close on the worker's lifelong connection; returns immediately.
        One at a time: False while a command is still executing (or the queue is full)."""
        if self._busy_v.value:
            return False
        try:
            self._cmd_q.put_nowait(dict(action=action, width=float(width), force=float(force)))
        except queue.Full:
            return False
        return True

    def stop(self):
        self._running_v.value = 0
        self._worker.join(timeout=1.5)        # a grasp may be in flight
        if self._worker.is_alive():
            self._worker.terminate()
