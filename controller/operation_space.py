import argparse
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from controller.base_controller import BaseController, ControllerConfig
from utils.control import log_pose_error, pose_error_norms
from utils.motion import CartesianTargetPlanner
from utils.transforms import compute_pose_error, create_frame_from_xyzrpy


def pseudoinverse(matrix: np.ndarray, epsilon: float = 2.5e-4) -> np.ndarray:
    """Match the Deoxys SVD-based pseudoinverse with a fixed singular-value cutoff."""
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    singular_values_inv = np.zeros(matrix.shape, dtype=float)
    for i, singular_value in enumerate(singular_values):
        if singular_value >= epsilon:
            singular_values_inv[i, i] = 1.0 / singular_value
    return vh.T @ singular_values_inv @ u.T


@dataclass
class OperationSpaceCfg:
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(control_mode="torque", auto_goto_home=True)
    )
    trajectory_duration: float = 8.0
    translation_error_threshold: float = 5e-3
    rotation_error_threshold: float = 1e-2
    base_gains: Sequence[float] = (30.0, 30.0, 30.0, 60.0, 60.0, 60.0)
    residual_mass_vec: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5)
    planner_duration: float = 3.0
    # time_fraction: float = 0.3 
    log_interval: float = 1.0
    task_mode: str = "full"


class OperationSpaceController(BaseController):
    def __init__(self, robot_ip: str, config: OperationSpaceCfg | None = None):
        self.cfg = config if config is not None else OperationSpaceCfg()
        super().__init__(robot_ip, self.cfg.controller)

        self._motion_gains = np.array(self.cfg.base_gains)
        self._damping_gains = 2.0 * np.sqrt(self._motion_gains)
        
        self._task_mode = self.cfg.task_mode.strip().lower()
        # NOTE: This has to be the linear interpolation, slerp for rotation. 
        self._target_planner = CartesianTargetPlanner(default_duration=self.cfg.planner_duration)
        
        self._planned_target = None
        self._goal_pose = None
        self.time_elapsed = 0.0
        self.motion_finished = False
        self._timed_out = False

        self._next_log_time = 0.0

    def task_error(self, current_pose, target_pose):
        return compute_pose_error(current_pose, target_pose)

    def _goal_reached(self, error_6d: np.ndarray) -> bool:
        translation_error, rotation_error, _ = pose_error_norms(error_6d)
        return (
            translation_error < self.cfg.translation_error_threshold
            and rotation_error < self.cfg.rotation_error_threshold
        )

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

        if self._task_mode == "translation_only":
            error_6d[3:] = 0.0
            eef_velocity[3:] = 0.0
        elif self._task_mode == "rotation_only":
            pass  # keep full position+orientation control; position is held via zero translation delta
        elif self._task_mode != "full":
            raise ValueError(
                f"Unsupported task_mode '{self.cfg.task_mode}'. "
                "Supported values are: full, translation_only, rotation_only."
            )

        pos_error = error_6d[:3]
        ori_error = error_6d[3:]
        linear_velocity = eef_velocity[:3]
        angular_velocity = eef_velocity[3:]

        des_acc_pos = self._motion_gains[:3] * pos_error - self._damping_gains[:3] * linear_velocity
        des_acc_ori = self._motion_gains[3:] * ori_error - self._damping_gains[3:] * angular_velocity

        jacobian_pos = jacobian[:3, :]
        jacobian_ori = jacobian[3:, :]

        mass_matrix_inv = np.linalg.inv(mass_matrix)
        lambda_pos_inv = jacobian_pos @ mass_matrix_inv @ jacobian_pos.T
        lambda_ori_inv = jacobian_ori @ mass_matrix_inv @ jacobian_ori.T
        lambda_pos = pseudoinverse(lambda_pos_inv)
        lambda_ori = pseudoinverse(lambda_ori_inv)

        task_wrench_pos = lambda_pos @ des_acc_pos
        task_wrench_ori = lambda_ori @ des_acc_ori  # used for logging only

        # Dynamically-consistent null-space projector for position task:
       
        null_projector = np.eye(7) - jacobian_pos.T @ lambda_pos @ jacobian_pos @ mass_matrix_inv

      
        tau_raw = (
            jacobian_pos.T @ task_wrench_pos
            + null_projector @ jacobian_ori.T @ des_acc_ori
            + self._coriolis
        )

        tau_rate_limited = self.apply_torque_rate_limit(tau_raw)
        tau_clipped = self.clip_torques(tau_rate_limited)
        return tau_clipped

    def _control_step(self) -> None:
        if self._target_planner.current_target is None:
            raise ValueError("Planner has no target. Call set_target(...) before run().")

        self._update_state()
        self.time_elapsed += self.duration.to_sec()
        self._planned_target = self._target_planner.step(self.duration.to_sec())

        tau_d = self._compute_torque()
        planned_error_6d = self.task_error(self._cartesian_pose, self._planned_target)
        final_error_6d = self.task_error(self._cartesian_pose, self._target_planner.goal_pose)
        planned_norms = pose_error_norms(planned_error_6d)
        final_norms = pose_error_norms(final_error_6d)

        planner_finished = self._target_planner.is_finished()
        reached_goal = planner_finished and self._goal_reached(final_error_6d)
        self._timed_out = self.time_elapsed >= self.cfg.trajectory_duration
        self.motion_finished = reached_goal or self._timed_out

        if self.time_elapsed >= self._next_log_time:
            log_pose_error(
                "planned_target_error",
                planned_norms,
                time_elapsed=self.time_elapsed,
                include_total=False,
            )
            log_pose_error(
                "goal_error",
                final_norms,
                time_elapsed=self.time_elapsed,
                include_total=False,
            )
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
        log_pose_error(
            "final_goal_error",
            final_norms,
            time_elapsed=self.time_elapsed,
            include_total=False,
        )
        return goal_pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--dx", type=float, default=0.0, help="X translation offset in meters")
    parser.add_argument("--dy", type=float, default=0.0, help="Y translation offset in meters")
    parser.add_argument("--dz", type=float, default=0.0, help="Z translation offset in meters")
    parser.add_argument("--droll", type=float, default=0.0, help="Roll rotation offset in degrees")
    parser.add_argument("--dpitch", type=float, default=0.0, help="Pitch rotation offset in degrees")
    parser.add_argument("--dyaw", type=float, default=0.0, help="Yaw rotation offset in degrees")
    parser.add_argument("--duration", type=float, default=None, help="Optional target interpolation duration")
    parser.add_argument(
        "--task-mode",
        type=str,
        default="full",
        choices=["full", "translation_only", "rotation_only"],
        help="Enable full 6D task, translation-only task, or rotation-only task.",
    )
    args = parser.parse_args()

    controller = None
    try:
        controller = OperationSpaceController(args.ip, OperationSpaceCfg(task_mode=args.task_mode))
        # NOTE: we are going to be setting the delta target for now. 
        delta_target = create_frame_from_xyzrpy(
            xyz=(args.dx, args.dy, args.dz),
            rpy=(
                np.deg2rad(args.droll),
                np.deg2rad(args.dpitch),
                np.deg2rad(args.dyaw),
            ),
        )
        controller.run(delta_target=delta_target, duration=args.duration)
    except Exception as exc:
        print(f"Error occurred: {exc}")
        if controller is not None:
            controller.robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    main()
