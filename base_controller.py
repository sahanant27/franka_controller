from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pylibfranka import ControllerMode, RealtimeConfig, Robot, Torques

from utils.control import franka_array_to_matrix, limit_torque_rate
from utils.motion import goto_pose


@dataclass
class ControllerConfig:
    """Single configuration object used to initialize a controller instance."""

    control_mode: str = "jpose"
    realtime_config: RealtimeConfig = RealtimeConfig.kIgnore # kIgnore | kEnforce
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
        residual_mass_vec = getattr(self, "cfg", None)
        residual_mass_vec = getattr(residual_mass_vec, "residual_mass_vec", None)
        if residual_mass_vec is None:
            self.residual_mass_matrix = np.zeros((7, 7))
        else:
            residual_mass_vec = np.asarray(residual_mass_vec, dtype=float)
            if residual_mass_vec.shape != (7,):
                raise ValueError(
                    "residual_mass_vec must contain 7 values for the Panda joint-space mass matrix."
                )
            self.residual_mass_matrix = np.diag(residual_mass_vec)
        self.robot_state = None
        self.duration = None

        self._set_collision_behavior()

        if self.config.auto_goto_home:
            self.goto_home()

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
            "Supported values are: torque, osc, jpose, jvel, cpose, cvel."
        )

    def goto_home(self) -> None:
        """Move robot to the project home configuration."""
        goto_pose(self.robot)

    def _update_state(self):
        """Read and cache one state sample; return (robot_state, duration)."""
        self.robot_state, self.duration = self.active_control.readOnce()
        return self.robot_state, self.duration

    ### Helper properties and methods for controller implementations ###
    @property
    def _joint_pose(self):
        """Return current 7-DOF joint position vector."""
        return np.array(self.robot_state.q)

    @property
    def _cartesian_pose(self):
        """Return current end-effector pose as a 4x4 matrix."""
        return franka_array_to_matrix(self.robot_state.O_T_EE, (4, 4))

    @property
    def _mass_matrix(self):
        """Return 7x7 joint-space mass matrix including any configured residual term."""
        mass_matrix = franka_array_to_matrix(self.model.mass(self.robot_state), (7, 7))
        return mass_matrix + self.residual_mass_matrix

    @property
    def _coriolis(self):
        """Return 7-DOF Coriolis forces."""
        return np.array(self.model.coriolis(self.robot_state))
    
    @property
    def _jacobian(self):
        """Return 6x7 end-effector Jacobian in base frame."""
        return franka_array_to_matrix(self.model.zero_jacobian(self.robot_state), (6, 7))

  
    ### Motion Generation Helper Methods ###
    ## Maybe a different class initself we will look into this later ##

    ### GOTO HOME Motion () """ 

    def task_error(self, current_pose, target_pose):
        """Compute controller-specific error between current and target pose."""
        raise NotImplementedError("task_error is intentionally left unimplemented.")
    
    def _motion_finished(self, error_norm) -> bool:
        """Determine if motion is finished based on error norm and elapsed time."""
        return error_norm < self.cfg.error_threshold or self.time_elapsed >= self.cfg.trajectory_duration
    
    def _joint_error(self, current_pose, target_pose):
        """Compute 7-DOF joint position error."""
        return target_pose - current_pose

    def apply_torque_rate_limit(self, tau_desired: np.ndarray) -> np.ndarray:
        """Apply per-joint torque rate limiting using configured max_delta_tau."""
        tau_reference = np.array(self.robot_state.tau_J_d)
        return limit_torque_rate(tau_desired, tau_reference, self.max_delta_tau)

    def clip_torques(self, tau: np.ndarray) -> np.ndarray:
        """Clip torques using configured per-joint limits."""
        return np.clip(tau, -self.max_torques, self.max_torques)

    def write_command(self, command_value, motion_finished: bool = False):
        """Write command to active control based on configured controller mode."""
        mode = self.config.control_mode.strip().lower()

        if mode in ["torque", "osc"]:
            values = np.asarray(command_value, dtype=float).tolist()
            command = Torques(values)
            command.motion_finished = motion_finished
            self.active_control.writeOnce(command)
            return command

        if mode == "jpose":
            raise NotImplementedError("write_command for jpose is not implemented yet.")
        if mode == "jvel":
            raise NotImplementedError("write_command for jvel is not implemented yet.")
        if mode == "cpose":
            raise NotImplementedError("write_command for cpose is not implemented yet.")
        if mode == "cvel":
            raise NotImplementedError("write_command for cvel is not implemented yet.")

        raise ValueError(
            f"Unsupported control_mode '{self.config.control_mode}' in write_command."
        )
