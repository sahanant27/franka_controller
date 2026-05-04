from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pylibfranka import ControllerMode, RealtimeConfig, Robot

from utils.control import franka_array_to_matrix


@dataclass
class ControllerConfig:
    """Single configuration object used to initialize a controller instance."""

    control_mode: str = "jpose"
    realtime_config: RealtimeConfig = RealtimeConfig.kIgnore  # kIgnore | kEnforce
    # Subclasses check this flag to decide whether to home at startup
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


class BaseController(ABC):
    """Common setup and state accessors for all Franka controllers.

    Subclass responsibilities:
      - goto_home(): implement homing appropriate for the control mode
      - write_command(): write the mode-specific command object via active_control
      - Call self.goto_home() in __init__ if self.config.auto_goto_home is True
    """

    def __init__(self, robot_ip: str, config: ControllerConfig | None = None):
        self.config = config if config is not None else ControllerConfig()
        self.robot = Robot(robot_ip, self.config.realtime_config)

        self.max_torques = np.array(self.config.max_torques)
        self.max_delta_tau = float(self.config.max_delta_tau)

        self.robot_state = None
        self.duration = None

        self._set_collision_behavior()

        # Subclasses call goto_home() here if auto_goto_home is True,
        # since the implementation differs per control mode.

        self.active_control = self._start_active_control(self.config.control_mode)
        self.model = self.robot.load_model()

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
            "Supported: torque, osc, jpose, jvel, cpose, cvel."
        )

    def _update_state(self):
        """Read and cache one state sample from active_control."""
        self.robot_state, self.duration = self.active_control.readOnce()
        return self.robot_state, self.duration

    # ------------------------------------------------------------------
    # State accessors — available once _update_state() has been called
    # ------------------------------------------------------------------

    @property
    def _joint_pose(self) -> np.ndarray:
        """Current 7-DOF joint position vector."""
        return np.array(self.robot_state.q)

    @property
    def _joint_velocity(self) -> np.ndarray:
        """Current 7-DOF joint velocity vector."""
        return np.array(self.robot_state.dq)

    @property
    def _cartesian_pose(self) -> np.ndarray:
        """Current end-effector pose as a 4x4 matrix (base frame)."""
        return franka_array_to_matrix(self.robot_state.O_T_EE, (4, 4))

    @property
    def _mass_matrix(self) -> np.ndarray:
        """7x7 joint-space mass matrix."""
        return franka_array_to_matrix(self.model.mass(self.robot_state), (7, 7))

    @property
    def _coriolis(self) -> np.ndarray:
        """7-DOF Coriolis and centrifugal forces."""
        return np.array(self.model.coriolis(self.robot_state))

    @property
    def _jacobian(self) -> np.ndarray:
        """6x7 end-effector Jacobian in base frame."""
        return franka_array_to_matrix(self.model.zero_jacobian(self.robot_state), (6, 7))

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------

    def _joint_error(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """7-DOF joint position error (target − current)."""
        return target_pose - current_pose

    def clip_torques(self, tau: np.ndarray) -> np.ndarray:
        """Clip torques to per-joint limits from config."""
        return np.clip(tau, -self.max_torques, self.max_torques)

    @abstractmethod
    def write_command(self, command_value, motion_finished: bool = False) -> None:
        """Write a command to active_control. Subclasses implement per mode."""
