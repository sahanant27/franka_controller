import argparse
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from base_controller import BaseController, ControllerConfig
from utils.control import log_pose_error, pose_error_norms
from utils.motion import CartesianTargetPlanner
from utils.transforms import compute_pose_error, create_frame_from_xyzrpy


@dataclass
class OperationSpaceCfg:
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(control_mode="torque", auto_goto_home=True)
    )
    trajectory_duration: float = 15.0
    error_threshold: float = 1e-3
    base_gains: Sequence[float] = (150.0, 150.0, 150.0, 450.0, 450.0, 450.0)
    planner_duration: float = 3.0
    log_interval: float = 1.0


class OperationSpaceController(BaseController):
    def __init__(self, robot_ip: str, config: OperationSpaceCfg | None = None):
        self.cfg = config if config is not None else OperationSpaceCfg()
        super().__init__(robot_ip, self.cfg.controller)

        self._motion_gains = np.array(self.cfg.base_gains)
        self._damping_gains = 2.0 * np.sqrt(self._motion_gains)
        self._target_planner = CartesianTargetPlanner(default_duration=self.cfg.planner_duration)
        self._planned_target = None
        self._goal_pose = None
        self.time_elapsed = 0.0
        self.motion_finished = False
        self._timed_out = False
        self._next_log_time = 0.0

    def task_error(self, current_pose, target_pose):
        return compute_pose_error(current_pose, target_pose)

    def set_target(self, abs_target=None, delta_target=None, duration=None) -> np.ndarray:
        self._update_state()
        goal_pose = self._target_planner.set_goal(
            current_pose=self._cartesian_pose,
            abs_target=abs_target,
            delta_target=delta_target,
            duration=duration,
        )
        self._goal_pose = goal_pose.copy()
        self._planned_target = self._target_planner.current_target.copy()
        self.motion_finished = False
        self.time_elapsed = 0.0
        self._timed_out = False
        self._next_log_time = 0.0
        return goal_pose

    def _compute_torque(self) -> np.ndarray:
        if self._planned_target is None:
            raise ValueError("Target is not set. Call set_target(...) before running control.")

        mass_matrix = self._mass_matrix.copy()
        jacobian = self._jacobian.copy()
        current_pose = self._cartesian_pose.copy()
        dq = np.array(self.robot_state.dq)
        eef_velocity = jacobian @ dq

        error_6d = self.task_error(current_pose, self._planned_target)
        des_acc = self._motion_gains * error_6d - self._damping_gains * eef_velocity

        mass_matrix_inv = np.linalg.inv(mass_matrix)
        lambda_inv = jacobian @ mass_matrix_inv @ jacobian.T
        tau_d = jacobian.T @ np.linalg.inv(lambda_inv) @ des_acc + self._coriolis

        tau_d = self.apply_torque_rate_limit(tau_d)
        tau_d = self.clip_torques(tau_d)
        return tau_d

    def _control_step(self) -> None:
        if self._target_planner.current_target is None:
            raise ValueError("Planner has no target. Call set_target(...) before run().")

        self._update_state()
        self.time_elapsed += self.duration.to_sec()
        self._planned_target = self._target_planner.step(self.duration.to_sec())

        tau_d = self._compute_torque()
        final_error_6d = self.task_error(self._cartesian_pose, self._target_planner.goal_pose)
        final_error_norm = np.linalg.norm(final_error_6d)
        final_norms = pose_error_norms(final_error_6d)

        planner_finished = self._target_planner.is_finished()
        reached_goal = planner_finished and final_error_norm < self.cfg.error_threshold
        self._timed_out = self.time_elapsed >= self.cfg.trajectory_duration
        self.motion_finished = reached_goal or self._timed_out

        if self.time_elapsed >= self._next_log_time:
            log_pose_error("goal_error", final_norms, time_elapsed=self.time_elapsed)
            self._next_log_time += self.cfg.log_interval

        self.write_command(tau_d, motion_finished=self.motion_finished)

    def run(self, abs_target=None, delta_target=None, duration=None) -> np.ndarray:
        goal_pose = self.set_target(abs_target=abs_target, delta_target=delta_target, duration=duration)
        print(f"Starting operation-space motion with timeout {self.cfg.trajectory_duration:.2f}s")
        while not self.motion_finished:
            self._control_step()
        final_error_6d = self.task_error(self._cartesian_pose, self._goal_pose)
        final_norms = pose_error_norms(final_error_6d)
        if self._timed_out:
            print(f"Motion timed out at t={self.time_elapsed:.2f}s")
        else:
            print(f"Motion finished at t={self.time_elapsed:.2f}s")
        log_pose_error("final_goal_error", final_norms, time_elapsed=self.time_elapsed)
        return goal_pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--dx", type=float, default=0.10, help="X translation offset in meters")
    parser.add_argument("--dy", type=float, default=0.0, help="Y translation offset in meters")
    parser.add_argument("--dz", type=float, default=0.0, help="Z translation offset in meters")
    parser.add_argument("--duration", type=float, default=None, help="Optional target interpolation duration")
    args = parser.parse_args()

    controller = None
    try:
        controller = OperationSpaceController(args.ip, OperationSpaceCfg())
        delta_target = create_frame_from_xyzrpy(xyz=(args.dx, args.dy, args.dz), rpy=(np.deg2rad(0.0), 0.0, 0.0))
        controller.run(delta_target=delta_target, duration=args.duration)
    except Exception as exc:
        print(f"Error occurred: {exc}")
        if controller is not None:
            controller.robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    main()
