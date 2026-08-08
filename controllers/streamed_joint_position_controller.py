#!/usr/bin/env python3
"""1 kHz streamed joint-position control — the executor for CONTINUOUS targets.

Why this exists: AsyncPositionControlHandler is a GOAL-SEEKER — every target is
treated as a final destination and the firmware decelerates to zero at it. Fed a
stream of targets (a policy at 10-25 Hz, or interpolated micro-steps) the arm
perpetually brakes-and-restarts: start-stop at sparse rates, judder at dense
ones. No client-side smoothing can fix that, because the braking is planned in
the firmware after the last thing we control.

Here WE generate the reference at 1 kHz and the firmware just tracks it
(readOnce -> writeOnce(JointPositions), same ActiveControl pattern as the
validated torque loop). A critically-damped 2nd-order tracker slides the
reference toward the latest streamed target with velocity continuity —
the reference never parks, so the arm never brakes between targets.

Same duck-type interface as TargetStreamer (update_target / get_state / stop /
_error), so franka_server can back its protocol with either executor.
Bonus: the state cache refreshes at the full 1 kHz.
"""
import threading

import numpy as np
import pylibfranka as franka

from controllers.joint_position_controller import Q_MIN, Q_MAX, _LO_TAU, _LO_F


def _start_position_control(robot):
    """The binding names the starter slightly differently across versions."""
    for name, args in (
        ("start_joint_position_control", (franka.ControllerMode.JointImpedance,)),
        ("start_joint_position_control", ()),
        ("start_joint_positions_control", ()),
    ):
        fn = getattr(robot, name, None)
        if fn is not None:
            try:
                return fn(*args)
            except TypeError:
                continue
    raise RuntimeError(
        "pylibfranka binding exposes no joint-position ActiveControl starter "
        f"(have: {[m for m in dir(robot) if m.startswith('start')]}). "
        "Use the async controller or the impedance server instead.")


class StreamedJointPositionController:
    """Owns the robot: 1 kHz reference-tracking loop in a background thread.

    update_target(q): thread-safe, any rate — the loop glides toward the latest.
    get_state():      {'q','dq','T_base_ee','controlling'} (+'error'), 1 kHz fresh.
    """

    def __init__(self, robot, tau=0.06, max_velocity=1.0, set_collision=True):
        self.robot = robot
        if set_collision:
            robot.set_collision_behavior(_LO_TAU, _LO_TAU, _LO_F, _LO_F)
        self._wn = 1.0 / float(tau)
        self.max_velocity = float(max_velocity)
        self._active = _start_position_control(robot)

        state, _ = self._active.readOnce()
        q0 = np.array(state.q)
        self._x = q0.copy()                      # reference position (what we write)
        self._v = np.zeros(7)                    # reference velocity
        self._lock = threading.Lock()
        self._target = q0.copy()                 # latest streamed target (hold pose)
        self._state = self._snapshot(state)
        self._error = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="streamed_pos_1khz")
        self._thread.start()

    @staticmethod
    def _snapshot(state):
        T = np.array(state.O_T_EE).reshape(4, 4, order="F")
        return {"q": np.array(state.q), "dq": np.array(state.dq), "T_base_ee": T}

    def update_target(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"target must have 7 elements, got {q.shape}")
        with self._lock:
            self._target = np.clip(q, Q_MIN, Q_MAX)

    def get_state(self):
        with self._lock:
            snap = dict(self._state)
            snap["controlling"] = self._running
            if self._error:
                snap["error"] = self._error
            return snap

    def _loop(self):
        # HOT PATH: must respond every 1 ms or the firmware reflexes with
        # communication_constraints_violation. Keep allocations minimal and the
        # lock window tiny; the state snapshot (np.array x3 + reshape + dict) is
        # the expensive part, so it runs decimated at 100 Hz — still 2-4x fresher
        # than anything the clients consume.
        read, write = self._active.readOnce, self._active.writeOnce
        JP = franka.JointPositions
        wn2, two_wn, vmax = self._wn * self._wn, 2.0 * self._wn, self.max_velocity
        count = 0
        try:
            while self._running:
                state, duration = read()
                dt = duration.to_sec()
                if not 0.0 < dt <= 0.01:                   # first call / hiccup guard
                    dt = 0.001
                count += 1
                if count % 10 == 0:                        # 100 Hz state cache
                    snap = self._snapshot(state)
                    with self._lock:
                        self._state = snap
                        goal = self._target
                else:
                    with self._lock:
                        goal = self._target
                # critically-damped tracker: reference glides, never parks
                acc = wn2 * (goal - self._x) - two_wn * self._v
                self._v = np.clip(self._v + acc * dt, -vmax, vmax)
                self._x = np.clip(self._x + self._v * dt, Q_MIN, Q_MAX)
                write(JP(self._x.tolist()))
        except Exception as e:                             # loop died: surface, stop
            self._error = str(e)
            self._running = False

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:                                               # finish the motion cleanly
            fin = franka.JointPositions(self._x.tolist())
            fin.motion_finished = True
            self._active.writeOnce(fin)
        except Exception:
            pass
