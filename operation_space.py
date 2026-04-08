import argparse
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pylibfranka import Torques

from base_controller import BaseController, ControllerConfig



@dataclass
class OperationSpaceCfg:
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(control_mode="torque", auto_goto_home=True)
    )
    decay: float = 0.995
    trajectory_duration: float = 15.0
    error_threshold: float = 1e-3
    base_gains: Sequence[float] = (150.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    translation_axis: str = "x"
    translation_distance: float = 0.10
    rotation_axis: str = "x"
    rotation_angle_deg: float = 0.0


class OperationSpaceController(BaseController):
    def __init__(self, robot_ip: str, config: OperationSpaceCfg | None = None):
        self.cfg = config if config is not None else OperationSpaceCfg()
        super().__init__(robot_ip, self.cfg.controller)

        self._motion_gains = None
        self._damping_gains = None
        self._target_frame = None

     

    def _set_target(self) -> None:
        pass


    # def _update_target(self) -> None:


    def _compute_torque(self, robot_state) -> np.ndarray:
        m = self.get_mass_matrix(robot_state)
        j = self.get_jacobian(robot_state)
        current_pose = self.get_cartesian_pose(robot_state)

        dq = np.array(robot_state.dq)
        eef_velocity = j @ dq

        error_6d = self.find_error(current_pose, self.target_frame)
        des_acc = self._motion_gains * error_6d - self._damping_gains * eef_velocity

        m_inv = np.linalg.inv(m)
        lambda_inv = j @ m_inv @ j.T
        tau_d = j.T @ np.linalg.inv(lambda_inv) @ des_acc

        tau_j_d = np.array(robot_state.tau_J_d)
        tau_d = self.apply_torque_rate_limit(tau_d, tau_j_d)
        tau_d = self.clip_torques(tau_d)
        return tau_d

    def step(self) -> None:
        robot_state, duration = self.read_state()
        self.time_elapsed += duration.to_sec()

        self._update_target()
        tau_d = self._compute_torque(robot_state)

        current_pose = self.get_cartesian_pose(robot_state)
        final_error_6d = self.find_error(current_pose, self.target_frame)
        final_error_norm = np.linalg.norm(final_error_6d)

        command = Torques(tau_d.tolist())
        command.motion_finished = (
            final_error_norm < self.cfg.error_threshold
            or self.time_elapsed >= self.cfg.trajectory_duration
        )
        self.motion_finished = command.motion_finished

        self.active_control.writeOnce(command)

    def run(self) -> None:
        while not self.motion_finished:
            self.step()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    controller = None
    try:
        controller = OperationSpaceController(args.ip, OperationSpaceCfg())
        controller.run()
    except Exception as exc:
        print(f"Error occurred: {exc}")
        if controller is not None:
            controller.robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    main()
