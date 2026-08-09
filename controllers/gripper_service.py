#!/usr/bin/env python3
"""Non-blocking gripper access shared by the servers.

NOTHING gripper-related may run on a thread of the control process:
pylibfranka's gripper calls hold the GIL — measured on this machine, a single
Gripper.read_once() can block 80 ms = 80 missed 1 kHz cycles = the firmware's
communication_constraints_violation reflex. So reads AND commands both run in
separate spawned processes (own GIL, own gripper connection):

  reads    — a persistent poll process caches width/is_grasped (~10 Hz) into
             shared memory; get_state attaches them for free.
  commands — fire-and-forget spawned process via home.set_gripper; the zmq
             reply returns immediately, mirroring the ROS stack's
             send_goal_async semantics.

The arm's control loop keeps running throughout — no stop/restart around
gripper motion.
"""
import multiprocessing
import time

from controllers.home import set_gripper


def _poll_gripper(ip, poll_s, width_v, grasped_v, running_v):
    """Poll process body (own GIL): cache gripper state into shared values."""
    import pylibfranka as franka
    g = franka.Gripper(ip)
    while running_v.value:
        try:
            gs = g.read_once()
            width_v.value = float(gs.width)
            grasped_v.value = int(gs.is_grasped)
        except Exception:
            pass                              # transient read hiccup: keep last value
        time.sleep(poll_s)


class GripperService:
    def __init__(self, ip, poll_s=0.1):
        self.ip = ip
        self._ctx = multiprocessing.get_context("spawn")   # never fork libfranka state
        self._width_v = self._ctx.Value("d", -1.0)         # < 0 = no read yet
        self._grasped_v = self._ctx.Value("i", -1)
        self._running_v = self._ctx.Value("i", 1)
        self._proc = None                                  # in-flight command process
        self._poller = self._ctx.Process(
            target=_poll_gripper, daemon=True,
            args=(ip, poll_s, self._width_v, self._grasped_v, self._running_v))
        self._poller.start()

    def wait_ready(self, timeout=5.0):
        """Block until the poll process has its first reading (= gripper connection is up).
        Arming the robot's RT session while the gripper child is still connecting trips
        communication_constraints_violation at session start (bisected 2026-08-08:
        settled service + 1 kHz loop = clean; connecting service + loop start = dead in
        0.5 s). Mid-session connects are tolerated — only session START is fragile."""
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
        """Fire-and-forget open/close in a spawned process. One at a time."""
        if self._proc is not None and self._proc.is_alive():
            return False
        self._proc = self._ctx.Process(
            target=set_gripper, daemon=True,
            kwargs=dict(ip=self.ip, action=action, width=float(width),
                        force=float(force)))
        self._proc.start()
        return True

    def stop(self):
        self._running_v.value = 0
        self._poller.join(timeout=0.5)
        if self._poller.is_alive():
            self._poller.terminate()
