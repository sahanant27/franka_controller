from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pylibfranka import ControllerMode, RealtimeConfig, Robot

from utils.control import franka_array_to_matrix, limit_torque_rate
from utils.motion import goto_pose


@dataclass
class ControllerConfig:
    """Single configuration object used to initialize a controller instance."""

    control_mode: str = "jpose"
    realtime_config: RealtimeConfig = RealtimeConfig.kIgnore
    auto_goto_home: bool = True
    lower_torque_thresholds: Sequence[float] = field(
        default_factory=lambda: (20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0)
    )
    upper_torque_thresholds: Sequence[float] = field(
        default_factory=lambda: (20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0)
    )
    lower_force_thresholds: Sequence[float] = field(
        default_factory=lambda: (20.0, 20.0, 20.0, 25.0, 25.0, 25.0)
    )
    upper_force_thresholds: Sequence[float] = field(
        default_factory=lambda: (20.0, 20.0, 20.0, 25.0, 25.0, 25.0)
    )
    max_torques: Sequence[float] = field(
        default_factory=lambda: (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    )
    max_delta_tau: float = 1.0


class BaseController:
    """Common setup and state helper class for Franka controllers."""

    def __init__(self, robot_ip: str, config: ControllerConfig | None = None):
        self.config = config if config is not None else ControllerConfig()
        self.robot = Robot(robot_ip, self.config.realtime_config)
        self.max_torques = np.array(self.config.max_torques)
        self.max_delta_tau = float(self.config.max_delta_tau)

        self._set_collision_behavior()
        self.active_control = self._start_active_control(self.config.control_mode)
        self.model = self.robot.load_model()

        if self.config.auto_goto_home:
            self.goto_home()

        self.robot_state = None
        self.duration = None

    def _set_collision_behavior(self) -> None:
        self.robot.set_collision_behavior(
            self.config.lower_torque_thresholds,
            self.config.upper_torque_thresholds,
            self.config.lower_force_thresholds,
            self.config.upper_force_thresholds,
        )

    def _start_active_control(self, control_mode: str):
        mode = control_mode.strip().lower()

        if mode in ["torque", "osc"]:
            return self.robot.start_torque_control()
        if mode == "jpose":
            return self.robot.start_joint_position_control(ControllerMode.JointImpedance)
        if mode == "jvel":
            return self.robot.start_joint_velocity_control(ControllerMode.JointImpedance)
        if mode == "cpose":
            return self.robot.start_cartesian_pose_control(ControllerMode.CartesianImpedance)
        if mode == "cvel":
            return self.robot.start_cartesian_velocity_control(ControllerMode.CartesianImpedance)

        raise ValueError(
            f"Unsupported control_mode '{control_mode}'. "
            "Supported values are: torque, osc, jpose, jvel, cpose, cvel."
        )

    def goto_home(self) -> None:
        """Move robot to the project home configuration."""
        goto_pose(self.robot)

    def curr_robot_state(self):
        """Read a single state sample from active control."""
        self.robot_state, self.duration = self.active_control.readOnce()
        return self.robot_state

    def read_state(self):
        """Read and cache one state sample; return (robot_state, duration)."""
        self.curr_robot_state()
        return self.robot_state, self.duration

    def _require_cached_state(self):
        if self.robot_state is None:
            raise RuntimeError("No cached robot state. Call read_state() once before accessing state properties.")
        return self.robot_state

    @property
    def _joint_pose(self):
        """Return current 7-DOF joint position vector."""
        return np.array(self._require_cached_state().q)

    @property
    def _cartesian_pose(self):
        """Return current end-effector pose as a 4x4 matrix."""
        return franka_array_to_matrix(self._require_cached_state().O_T_EE, (4, 4))

    @property
    def _mass_matrix(self):
        """Return 7x7 joint-space mass matrix."""
        return franka_array_to_matrix(self.model.mass(self._require_cached_state()), (7, 7))

    @property
    def _jacobian(self):
        """Return 6x7 end-effector Jacobian in base frame."""
        return franka_array_to_matrix(self.model.zero_jacobian(self._require_cached_state()), (6, 7))

    def get_joint_pose(self, robot_state=None) -> np.ndarray:
        state = self._require_cached_state() if robot_state is None else robot_state
        return np.array(state.q)

    def get_cartesian_pose(self, robot_state=None) -> np.ndarray:
        state = self._require_cached_state() if robot_state is None else robot_state
        return franka_array_to_matrix(state.O_T_EE, (4, 4))

    def get_mass_matrix(self, robot_state=None) -> np.ndarray:
        state = self._require_cached_state() if robot_state is None else robot_state
        return franka_array_to_matrix(self.model.mass(state), (7, 7))

    def get_jacobian(self, robot_state=None) -> np.ndarray:
        state = self._require_cached_state() if robot_state is None else robot_state
        return franka_array_to_matrix(self.model.zero_jacobian(state), (6, 7))

    def find_error(self, current_pose, target_pose):
        """Compute controller-specific error between current and target pose."""
        raise NotImplementedError("find_error is intentionally left unimplemented.")

    def apply_torque_rate_limit(self, tau_desired: np.ndarray, tau_reference: np.ndarray) -> np.ndarray:
        """Apply per-joint torque rate limiting using configured max_delta_tau."""
        return limit_torque_rate(tau_desired, tau_reference, self.max_delta_tau)

    def clip_torques(self, tau: np.ndarray) -> np.ndarray:
        """Clip torques using configured per-joint limits."""
        return np.clip(tau, -self.max_torques, self.max_torques)
    
    def control_loop(self):
        """Main control loop to be implemented by subclasses."""
        raise NotImplementedError("control_loop is intentionally left unimplemented.")


# Backward-compatible alias for older imports.
Controller = BaseController
