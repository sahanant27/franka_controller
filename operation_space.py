import argparse
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from base_controller import BaseController, ControllerConfig
from utils.transforms import compute_pose_error


@dataclass
class OperationSpaceCfg:
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(control_mode="torque", auto_goto_home=True)
    )
    trajectory_duration: float = 15.0
    error_threshold: float = 1e-3
    base_gains: Sequence[float] = (150.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    goto_pose : bool = False


class OperationSpaceController(BaseController):
    def __init__(self, robot_ip: str, config: OperationSpaceCfg | None = None):
        self.cfg = config if config is not None else OperationSpaceCfg()
        super().__init__(robot_ip, self.cfg.controller)

        self._motion_gains = np.array(self.cfg.base_gains)
        self._damping_gains = 2.0 * np.sqrt(self._motion_gains)
        self._target_frame = None
        self.time_elapsed = 0.0
        self.motion_finished = False

    def set_target(self, target_pose: np.ndarray) -> None:
        self._target_frame = target_pose

    def task_error(self, current_pose, target_pose):
        return compute_pose_error(current_pose, target_pose)

    def _compute_torque(self) -> np.ndarray:
        m = self._mass_matrix.copy()
        j = self._jacobian.copy()
        current_pose = self._cartesian_pose.copy()
        dq = self.robot_state.dq
        eef_velocity = j @ dq

        error_6d = self.task_error(current_pose, self._target_frame)
        des_acc = self._motion_gains * error_6d - self._damping_gains * eef_velocity

        m_inv = np.linalg.inv(m)
        lambda_inv = j @ m_inv @ j.T
        tau_d = j.T @ np.linalg.inv(lambda_inv) @ des_acc

        tau_d = self.apply_torque_rate_limit(tau_d)
        tau_d = self.clip_torques(tau_d)
        return tau_d

    def _control_step(self) -> None:
        if self._target_frame is None:
            raise ValueError("Target is not set. Call set_target(target_pose) before run().")

        self.time_elapsed += self.duration.to_sec()
        tau_d = self._compute_torque()

        current_pose = self._cartesian_pose.copy()
        final_error_6d = self.task_error(current_pose, self._target_frame)
        final_error_norm = np.linalg.norm(final_error_6d)

        self.motion_finished = self._motion_finished(final_error_norm)
        self.write_command(tau_d, motion_finished=self.motion_finished)

    def _should_continue_control(self, decimation_step: int, decimation: int) -> bool:
        if self.cfg.goto_pose:
            return not self.motion_finished
        return decimation_step < decimation

    def step(self, action, decimation: int = 1):
        if not self.cfg.goto_pose and decimation <= 0:
            raise ValueError("decimation must be a positive integer when goto_pose is False.")

        self.motion_finished = False
        self.time_elapsed = 0.0
        self.set_target(action)
        self._update_state()

        decimation_step = 0
        while self._should_continue_control(decimation_step, decimation):
            if decimation_step > 0:
                self._update_state()
            self._control_step()
            decimation_step += 1

    def run(self, target_pose: np.ndarray, decimation: int = 1) -> None:
        self.step(target_pose, decimation=decimation)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--dx", type=float, default=0.10, help="X translation offset in meters")
    parser.add_argument("--dy", type=float, default=0.0, help="Y translation offset in meters")
    parser.add_argument("--dz", type=float, default=0.0, help="Z translation offset in meters")
    args = parser.parse_args()

    controller = None
    try:
        controller = OperationSpaceController(args.ip, OperationSpaceCfg())
        controller._update_state()
        target_pose = controller._cartesian_pose.copy()
        target_pose[0, 3] += args.dx
        target_pose[1, 3] += args.dy
        target_pose[2, 3] += args.dz
        controller.run(target_pose)
    except Exception as exc:
        print(f"Error occurred: {exc}")
        if controller is not None:
            controller.robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    main()
