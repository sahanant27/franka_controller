#!/usr/bin/env python3
"""STEP 2 — reusable async joint-position controller.

Wraps the AsyncPositionControlHandler path (validated in async_joint_move.py) as
a class so the zmq server and calibration can both drive it. The 1 kHz tracking
runs inside libfranka; we only set joint targets and read state.

Notes baked in from Step 1:
  - feedback.status is an unregistered enum in this build -> never read it.
  - robot.read_once() works during async control -> completion is position-based.

Two ways to command:
  - move_to(q): blocking, returns when within tolerance (scripted moves, calibration).
  - set_target(q): non-blocking single command (streaming / policy / zmq).

CLI self-test (ROBOT PC, e-stop in hand):
    python joint_position_controller.py --ip 172.16.0.2 --joint 6 --delta 0.2 --hz 10
"""
import argparse
import threading
import time

import numpy as np
import pylibfranka as franka

# Franka Panda joint limits [rad] (FR3 differs slightly — adjust if needed).
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# Conservative collision thresholds (contact above these stops the arm).
_LO_TAU = [20, 20, 18, 18, 16, 14, 12]
_LO_F = [20, 20, 20, 25, 25, 25]


class JointPositionController:
    """Async joint-position control for a connected pylibfranka Robot."""

    def __init__(self, robot, max_velocity=0.4, goal_tolerance=0.05,
                 set_collision=True):
        self.robot = robot
        self.tol = goal_tolerance
        self.max_velocity = max_velocity
        if set_collision:
            robot.set_collision_behavior(_LO_TAU, _LO_TAU, _LO_F, _LO_F)
        cfg = franka.AsyncPositionControlHandler.Configuration(
            maximum_joint_velocities=[max_velocity] * 7, goal_tolerance=goal_tolerance)
        result = franka.AsyncPositionControlHandler.configure(robot, cfg)
        if result.error_message:
            raise RuntimeError(f"configure failed: {result.error_message}")
        self.handler = result.handler          # arm is now actively held at current pose

    # --- state -------------------------------------------------------------
    def read_q(self):
        return np.array(self.robot.read_once().q)

    def read_state(self):
        """(q, dq, T_base_ee 4x4) — for calibration / zmq get_state."""
        s = self.robot.read_once()
        T = np.array(s.O_T_EE).reshape(4, 4, order="F")   # libfranka is column-major
        return np.array(s.q), np.array(s.dq), T

    # --- commands ----------------------------------------------------------
    def _clamp(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"target q must have 7 elements, got {q.shape}")
        cl = np.clip(q, Q_MIN, Q_MAX)
        if not np.allclose(cl, q):
            print(f"WARNING: target clamped to joint limits: {q.round(3)} -> {cl.round(3)}")
        return cl

    def set_target(self, q):
        """Non-blocking: command one (clamped) joint target. Returns the sent target."""
        # ------------------------------------------------------------------
        q = self._clamp(q)
        tgt = franka.AsyncPositionControlHandler.JointPositionTarget(joint_positions=q.tolist())
        cmd = self.handler.set_joint_position_target(tgt)
        if cmd.error_message:
            raise RuntimeError(f"set_target error: {cmd.error_message}")
        return q

    def move_to(self, q_target, hz=50.0, timeout=None):
        """Blocking move. Returns (reached: bool, final_q). Re-sends + polls q."""
        q_target = self._clamp(q_target)
        q0 = self.read_q()
        if timeout is None:                  # distance / speed + settle margin
            timeout = float(np.max(np.abs(q_target - q0)) / self.max_velocity) + 3.0
        dt = 1.0 / hz
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            loop = time.monotonic()
            self.set_target(q_target)
            if np.max(np.abs(self.read_q() - q_target)) <= self.tol:
                return True, self.read_q()
            sleep = dt - (time.monotonic() - loop)
            if sleep > 0:
                time.sleep(sleep)
        return False, self.read_q()          # timed out before reaching tolerance

    def stop(self):
        self.handler.stop_control()


class TargetStreamer:
    """50 Hz feeder thread for streaming / policy control (SKETCH — untested on hw).

    Why this exists (libfranka source trace): set_target sends ONE UDP packet and
    libfranka never repeats it. The robot firmware drives toward the last target,
    but the reference design streams at 50 Hz and a 10 Hz policy is far sparser
    than anything Franka validates. So we re-send the latest target at a steady
    50 Hz here, while the policy updates that target at 10 Hz.

    This thread is the ONLY one that touches the robot (set_target + read_state);
    callers use update_target()/get_state() through a lock — mirroring the
    franka_zmq_server shared-state pattern. Intended to back the zmq server:
        get_state  -> streamer.get_state()
        set_target -> streamer.update_target(q)
    """

    def __init__(self, controller: JointPositionController, rate_hz=50.0):
        self.ctrl = controller
        self.dt = 1.0 / rate_hz
        self._lock = threading.Lock()
        q, dq, T = controller.read_state()
        self._target = q.copy()                  # start by holding the current pose
        self._state = {"q": q, "dq": dq, "T_base_ee": T}
        self._running = False
        self._thread = None
        self._error = None                       # control error surfaced from the thread

    def update_target(self, q):
        """Thread-safe: set the latest joint target (policy calls this ~10 Hz)."""
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"target must have 7 elements, got {q.shape}")
        with self._lock:
            self._target = q.copy()

    def get_state(self):
        """Thread-safe snapshot: {'q','dq','T_base_ee'} (+ 'error' if the loop died)."""
        with self._lock:
            snap = dict(self._state)
            if self._error:
                snap["error"] = self._error
            return snap

    def _loop(self):
        # Top-level thread boundary: record control errors instead of dying silently.
        try:
            while self._running:
                t = time.monotonic()
                with self._lock:
                    tgt = self._target.copy()
                self.ctrl.set_target(tgt)        # the re-sent UDP packet (keeps control alive)
                q, dq, T = self.ctrl.read_state()
                with self._lock:
                    self._state = {"q": q, "dq": dq, "T_base_ee": T}
                sleep = self.dt - (time.monotonic() - t)
                if sleep > 0:
                    time.sleep(sleep)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="target_streamer")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.ctrl.stop()                         # release async control


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--joint", type=int, default=6, help="which joint to nudge (0-6)")
    ap.add_argument("--delta", type=float, default=0.2, help="how far to move it [rad]")
    ap.add_argument("--max-vel", type=float, default=0.4)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    q0 = np.array(robot.read_once().q)
    target = q0.copy()
    target[args.joint] += args.delta
    print(f"start  q = {q0.round(3).tolist()}")
    print(f"target q = {target.round(3).tolist()}  (joint {args.joint} += {args.delta})")
    input("Workspace clear, e-stop in hand? Press Enter to move... ")

    ctrl = JointPositionController(robot, max_velocity=args.max_vel, goal_tolerance=args.tol)
    try:
        reached, qf = ctrl.move_to(target, hz=args.hz)
    finally:
        ctrl.stop()
    print(f"final  q = {qf.round(3).tolist()}")
    print(f"reached={reached}  joint {args.joint} moved {qf[args.joint]-q0[args.joint]:+.3f} "
          f"(wanted {args.delta:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
