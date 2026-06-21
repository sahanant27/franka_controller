#!/usr/bin/env python3
"""Joint-impedance controller driven by a 21-D action [Δq(7), Kp(7), Kd(7)].

This is the manipulation-policy controller. A background thread runs the 1 kHz
torque loop (pylibfranka start_torque_control); the policy pushes a new 21-D
action at ~10 Hz and the loop holds it in between.

Control law each tick (q, dq measured):
    tau = Kp * (q_ref - q) - Kd * dq + coriolis
The robot compensates gravity internally for torque control, so we add only
coriolis (matches the franka examples).

TODO — apply before any manipulation-policy run. The training law is RESOLVED and the
current code does NOT match it yet (sources agree: CORN pkm/scripts/real/controller.py:768,
the IsaacLab trace in the policy bundle's control_law.py, and grasping_ws/scripts/robot/
sim_impedance_ema.py):
  [ ] 1. Kd is a COEFFICIENT on sqrt(Kp), not an absolute gain. In set_action:
            self._kd = kd * np.sqrt(kp)          # currently: self._kd = kd.copy()
         As-is, the policy's Kd in [0.3,2.0] is used as absolute -> the arm RINGS (sim-proven).
  [ ] 2. PURE PD — drop coriolis to match training (the robot adds gravity in torque mode):
            tau = kp*(q_ref - q) - kd*dq          # currently: ... + coriolis
  [x] 3. Clamp the latched q_ref to FR3 soft limits (rel_clamp ±0.9 of range) — DONE in set_action.
  [ ] 4. (grasp, later) gripper open/close via franka.Gripper — not implemented anywhere yet.
  [ ] 5. (later) atomic position<->impedance mode switch for grasp <-> manipulation.

Δq is relative to the MEASURED q at action time, latched until the next action
(q_ref = q_meas + Δq; reference_mode="commanded" integrates onto the previous target instead).

Run on the ROBOT PC (pylibfranka env), e-stop in hand:
    python controllers/joint_impedance_controller.py --ip 172.16.0.2 --joint 6 --dq 0.1
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root for sibling controllers
from controllers.joint_position_controller import Q_MIN, Q_MAX   # FR3 joint limits (single source of truth)

# Panda per-joint torque limits [Nm].
MAX_TORQUES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
# Default hold gains (used until the policy sends an action). Policy overrides.
DEFAULT_KP = np.array([80.0, 80.0, 80.0, 80.0, 30.0, 20.0, 12.0])
DEFAULT_KD = 2.0 * np.sqrt(DEFAULT_KP)
# Absolute joint-target SOFT limits. The training env clamps target_q to rel_clamp_joint_target = [-0.9, 0.9]
_Q_MID = 0.5 * (Q_MIN + Q_MAX)
_Q_HALF = 0.5 * (Q_MAX - Q_MIN)
Q_SOFT_LO = _Q_MID - 0.9 * _Q_HALF
Q_SOFT_HI = _Q_MID + 0.9 * _Q_HALF

_LO_TAU = [100.0] * 7        # collision thresholds (loose: don't false-trip the policy)
_LO_F = [100.0] * 6


class JointImpedanceController:
    """1 kHz joint-impedance torque loop fed by a 21-D action."""

    def __init__(self, robot, max_delta_tau=1.0, reference_mode="measured", max_dq=None):
        self.robot = robot
        self.max_delta_tau = max_delta_tau          # per-tick torque slew limit [Nm]
        self.reference_mode = reference_mode        # "measured" | "commanded"
        self.max_dq = max_dq                        # optional |Δq| clamp per action [rad]
        #   (golden franka_zmq_server clamps the per-command target delta to 0.1;
        #    set this to your policy's safe Δq bound. None = trust the policy.)
        robot.set_collision_behavior(_LO_TAU, _LO_TAU, _LO_F, _LO_F)

        s = robot.read_once()
        q0 = np.array(s.q)
        self._lock = threading.Lock()
        self._q_ref = q0.copy()                     # latched target
        self._kp = DEFAULT_KP.copy()
        self._kd = DEFAULT_KD.copy()
        self._last_q = q0.copy()                    # measured, updated by the loop
        self._last_dq = np.zeros(7)
        self._ee = np.array(s.O_T_EE).reshape(4, 4, order="F")
        self._jac = np.zeros((6, 7))                # base-frame zero-Jacobian (∂[v;ω]_ee/∂q), latched by the loop
        self._prev_tau = np.zeros(7)
        self._running = False
        self._thread = None
        self._error = None

    # --- policy interface --------------------------------------------------
    def set_action(self, action):
        """Apply a 21-D action [Δq(7), Kp(7), Kd(7)]. Called by the policy ~10 Hz."""
        a = np.asarray(action, dtype=float)
        if a.shape != (21,):
            raise ValueError(f"action must be 21-D, got {a.shape}")
        dq, kp, kd = a[:7], a[7:14], a[14:21]
        if self.max_dq is not None:                 # safety clamp (golden does this on the target delta)
            dq = np.clip(dq, -self.max_dq, self.max_dq)
        with self._lock:
            base = self._last_q if self.reference_mode == "measured" else self._q_ref
            self._q_ref = np.clip(base + dq, Q_SOFT_LO, Q_SOFT_HI)
            self._kp = kp.copy()
            self._kd = kd.copy() * np.sqrt(kp)  



    def get_state(self):
        with self._lock:
            return {"q": self._last_q.copy(), "dq": self._last_dq.copy(),
                    "T_base_ee": self._ee.copy(), "jacobian": self._jac.copy(),
                    "controlling": self._running and self._error is None}

    # --- control loop ------------------------------------------------------
    def _loop(self):
        active = self.robot.start_torque_control()
        model = self.robot.load_model()
        try:
            while self._running:
                state, _ = active.readOnce()        # blocks ~1 ms (paces the loop)
                q = np.array(state.q)
                dq = np.array(state.dq)
                coriolis = np.array(model.coriolis(state))
                ee = np.array(state.O_T_EE).reshape(4, 4, order="F")
                jac = np.array(model.zero_jacobian(state)).reshape(6, 7, order="F")  # base frame, column-major
                with self._lock:
                    q_ref, kp, kd = self._q_ref.copy(), self._kp.copy(), self._kd.copy()
                    self._last_q, self._last_dq, self._ee = q, dq, ee
                    self._jac = jac

                tau = kp * (q_ref - q) - kd * dq 

                # safety: per-tick slew limit, then absolute clip
                tau = self._prev_tau + np.clip(tau - self._prev_tau,
                                               -self.max_delta_tau, self.max_delta_tau)
                tau = np.clip(tau, -MAX_TORQUES, MAX_TORQUES)
                self._prev_tau = tau

                cmd = franka.Torques(tau.tolist())
                cmd.motion_finished = False
                active.writeOnce(cmd)
        except Exception as e:                      # control thread boundary
            with self._lock:
                self._error = str(e)
            self._running = False
        finally:
            try:
                cmd = franka.Torques(self._prev_tau.tolist())
                cmd.motion_finished = True
                active.writeOnce(cmd)
            except Exception:
                pass

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="joint_impedance")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--joint", type=int, default=6, help="joint to nudge in the test")
    ap.add_argument("--dq", type=float, default=0.1, help="delta joint position [rad]")
    ap.add_argument("--kp", type=float, default=80.0, help="test stiffness for that joint")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold the action")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    ctrl = JointImpedanceController(robot)
    print(f"start q = {ctrl.get_state()['q'].round(3).tolist()}")
    input("Workspace clear, e-stop in hand? Press Enter to start impedance hold... ")
    ctrl.start()
    time.sleep(0.5)
    print("holding current pose (Δq=0). Sending a small Δq action...")

    # build a 21-D action: Δq on one joint, Kp/Kd for all
    action = np.zeros(21)
    action[args.joint] = args.dq                    # Δq
    action[7:14] = DEFAULT_KP
    action[7 + args.joint] = args.kp
    action[14:21] = 2.0 * np.sqrt(action[7:14])     # Kd = 2*sqrt(Kp)
    ctrl.set_action(action)

    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.hold:
            st = ctrl.get_state()
            print(f"  q[{args.joint}]={st['q'][args.joint]:+.3f}  controlling={st['controlling']}")
            time.sleep(0.5)
    finally:
        ctrl.stop()
    print(f"final q = {ctrl.get_state()['q'].round(3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
