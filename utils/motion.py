

import time
import numpy as np
from pylibfranka import Robot, Torques


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