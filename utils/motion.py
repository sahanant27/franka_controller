

import time
import numpy as np
from pylibfranka import Robot, Torques

from .transforms import slerp_rot_matrix


class SimpleMotionGenerator:
    """Simple minimum jerk trajectory generator for smooth joint motion."""

    def __init__(self, start_position, end_position, duration=3.0):
        """Initialize the trajectory generator.

        Args:
            start_position: Starting joint positions (array of 7 values)
            end_position: Target joint positions (array of 7 values)
            duration: Duration of the trajectory in seconds
        """
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.duration = duration
        self.start_time = None

    def start(self):
        """Start the trajectory."""
        self.start_time = time.time()

    def get_position(self):
        """Get the current target position along the trajectory."""
        if self.start_time is None:
            return self.start_position

        elapsed_time = time.time() - self.start_time
        s = self._minimum_jerk(min(elapsed_time / self.duration, 1.0))

        return self.start_position + s * (self.end_position - self.start_position)

    def is_finished(self):
        """Check if the trajectory is complete."""
        if self.start_time is None:
            return False

        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.duration

    def _minimum_jerk(self, t):
        """Minimum jerk trajectory profile (normalized [0,1])."""
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)


class CartesianTargetPlanner:
    """Reusable Cartesian target planner for absolute and delta pose goals."""

    def __init__(self, default_duration=3.0):
        self.default_duration = float(default_duration)
        self.reset()

    def reset(self):
        self.start_pose = None
        self.goal_pose = None
        self.current_target = None
        self.duration = self.default_duration
        self.elapsed = 0.0
        self.active = False

    def set_goal(self, current_pose, abs_target=None, delta_target=None, duration=None):
        if (abs_target is None) == (delta_target is None):
            raise ValueError("Exactly one of abs_target or delta_target must be provided.")

        self.start_pose = np.array(current_pose, copy=True)
        # NOTE: Do we give this the autonoy or the target?? should be just the target i feel.
        self.goal_pose = (
            np.array(abs_target, copy=True)
            if abs_target is not None
            else self.start_pose @ np.array(delta_target, copy=True)
        )
        self.current_target = self.start_pose.copy()
        self.duration = self.default_duration if duration is None else float(duration)
        self.duration = max(self.duration, 1e-6)
        self.elapsed = 0.0
        self.active = True
        return self.goal_pose

    def is_finished(self):
        return not self.active
    
    #NOTE: What is this used for???? 
    def _minimum_jerk(self, t):
        t = np.clip(t, 0.0, 1.0)
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)

    def step(self, dt):
        if self.current_target is None:
            raise ValueError("Planner has no goal. Call set_goal(...) first.")
        if not self.active:
            return self.goal_pose.copy()

        self.elapsed = min(self.elapsed + float(dt), self.duration)
        fraction = self._minimum_jerk(self.elapsed / self.duration)

        target = np.eye(4)
        target[:3, 3] = self.start_pose[:3, 3] + fraction * (
            self.goal_pose[:3, 3] - self.start_pose[:3, 3]
        )
        target[:3, :3] = slerp_rot_matrix(
            self.start_pose[:3, :3],
            self.goal_pose[:3, :3],
            fraction,
        )

        self.current_target = target
        if self.elapsed >= self.duration:
            self.active = False
            self.current_target = self.goal_pose.copy()

        return self.current_target.copy()


class JointTargetPlanner:
    """Minimum-jerk joint-space target planner. Mirrors CartesianTargetPlanner interface."""

    def __init__(self, default_duration: float = 3.0):
        self.default_duration = float(default_duration)
        self.reset()

    def reset(self):
        self.start_q = None
        self.goal_q = None
        self.current_target = None
        self.duration = self.default_duration
        self.elapsed = 0.0
        self.active = False

    def set_goal(self, current_q, target_q, duration=None):
        self.start_q = np.array(current_q, copy=True)
        self.goal_q = np.array(target_q, copy=True)
        self.current_target = self.start_q.copy()
        self.duration = self.default_duration if duration is None else float(duration)
        self.duration = max(self.duration, 1e-6)
        self.elapsed = 0.0
        self.active = True
        return self.goal_q

    def is_finished(self):
        return not self.active

    def _minimum_jerk(self, t):
        t = np.clip(t, 0.0, 1.0)
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)

    def step(self, dt):
        if self.current_target is None:
            raise ValueError("Planner has no goal. Call set_goal(...) first.")
        if not self.active:
            return self.goal_q.copy()

        self.elapsed = min(self.elapsed + float(dt), self.duration)
        fraction = self._minimum_jerk(self.elapsed / self.duration)
        self.current_target = self.start_q + fraction * (self.goal_q - self.start_q)

        if self.elapsed >= self.duration:
            self.active = False
            self.current_target = self.goal_q.copy()

        return self.current_target.copy()



def goto_pose(
    robot,
    target_joint_position=None,
    duration=3.0,
    joint_stiffness=None,
    joint_position_tolerance=1e-3,
    joint_velocity_tolerance=5e-3,
    settle_time=0.2,
    max_run_time=None,
):
    """Move to a joint target and exit only after strict full-joint convergence.

    Convergence requires all joints to satisfy both position and velocity
    tolerances continuously for ``settle_time`` seconds.
    """
    if target_joint_position is None:
        target_joint_position = [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]
    if joint_stiffness is None:
        joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    if max_run_time is None:
        max_run_time = duration + 5.0

    target_joint_position = np.array(target_joint_position)
    joint_stiffness = np.array(joint_stiffness)
    joint_damping = 2.0 * np.sqrt(joint_stiffness)

    # Get initial state
    initial_state = robot.read_once()
    current_position = np.array(initial_state.q)

    # Start torque control
    active_control = robot.start_torque_control()

    # Create a model instance from robot
    model = robot.load_model()

    trajectory = SimpleMotionGenerator(
        current_position,
        target_joint_position,
        duration=duration,
    )
    trajectory.start()

    start_time = time.time()
    converged_since = None

    while True:
        # Read robot state
        robot_state, _ = active_control.readOnce()

        # Get state variables
        coriolis = np.array(model.coriolis(robot_state))
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)

        # Get current target from trajectory
        q_goal = trajectory.get_position()

        # Compute error to desired equilibrium joint configuration
        position_error = q - q_goal

        # Compute joint-space impedance control
        tau_task = -joint_stiffness * position_error - joint_damping * dq

        # Add coriolis compensation
        tau_d = tau_task + coriolis

        # Convert to array for Torques command
        torque_command = Torques(tau_d.tolist())
        torque_command.motion_finished = False
        active_control.writeOnce(torque_command)

        # Strict convergence to final joint target (all joints, no partial completion).
        position_error_to_target = target_joint_position - q
        all_pos_converged = np.all(np.abs(position_error_to_target) <= joint_position_tolerance)
        all_vel_converged = np.all(np.abs(dq) <= joint_velocity_tolerance)

        now = time.time()
        if all_pos_converged and all_vel_converged:
            if converged_since is None:
                converged_since = now
            if now - converged_since >= settle_time:
                torque_command.motion_finished = True
                active_control.writeOnce(torque_command)
                break
        else:
            converged_since = None

        if now - start_time >= max_run_time:
            torque_command.motion_finished = True
            active_control.writeOnce(torque_command)
            break
