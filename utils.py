import numpy as np


def matrix_to_rpy(rotation_matrix):
    """Convert 3x3 rotation matrix to roll, pitch, yaw (Euler angles in radians).
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Extract rotation matrix from 4x4 transformation matrix if needed
    if rotation_matrix.shape == (4, 4):
        R = rotation_matrix[:3, :3]
    else:
        R = rotation_matrix
    
    # Calculate pitch
    sin_pitch = -R[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    
    # Calculate roll and yaw
    cos_pitch = np.cos(pitch)
    
    if np.abs(cos_pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    
    return roll, pitch, yaw


def matrix_to_quaternion(transformation_matrix):
    """Convert 4x4 transformation matrix to quaternion (wxyz format).
    
    Args:
        transformation_matrix: 4x4 transformation matrix
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    R = transformation_matrix[:3, :3]
    
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def create_target_frame_translation_only(position):
    """Create a target frame with only translation (identity rotation).
    
    Args:
        position: Translation vector [x, y, z]
        
    Returns:
        4x4 transformation matrix with identity rotation and given translation
    """
    T = np.eye(4)
    T[0, 3] = position[0]
    T[1, 3] = position[1]
    T[2, 3] = position[2]
    return T


def _axis_to_index(axis):
    """Map axis specifier ('x'/'y'/'z' or 0/1/2) to an integer index."""
    if isinstance(axis, str):
        axis = axis.lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError("axis must be one of: 'x', 'y', 'z', 0, 1, 2")
        return axis_map[axis]

    if axis in (0, 1, 2):
        return axis

    raise ValueError("axis must be one of: 'x', 'y', 'z', 0, 1, 2")


def create_delta_frame_translation_axis(axis, distance):
    """Create a 4x4 delta transform with translation only along one axis.

    Args:
        axis: Axis specifier ('x', 'y', 'z' or 0, 1, 2)
        distance: Translation amount in meters

    Returns:
        4x4 homogeneous transform representing the translation delta
    """
    axis_idx = _axis_to_index(axis)
    T_delta = np.eye(4)
    T_delta[axis_idx, 3] = distance
    return T_delta


def create_delta_frame_rotation_axis(axis, angle_rad):
    """Create a 4x4 delta transform with rotation only about one axis.

    Args:
        axis: Axis specifier ('x', 'y', 'z' or 0, 1, 2)
        angle_rad: Rotation amount in radians

    Returns:
        4x4 homogeneous transform representing the rotation delta
    """
    axis_idx = _axis_to_index(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    R = np.eye(3)
    if axis_idx == 0:
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis_idx == 1:
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    else:
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    T_delta = np.eye(4)
    T_delta[:3, :3] = R
    return T_delta


def compute_6d_error(current_pose, target_pose):
    """Compute 6D error (position and orientation) for OSC control.
    
    Args:
        current_pose: Current 4x4 transformation matrix
        target_pose: Target 4x4 transformation matrix
        
    Returns:
        6D error vector [position_error_x, position_error_y, position_error_z, 
                         orientation_error_x, orientation_error_y, orientation_error_z]
    """
    # Position error (3D)
    position_error = target_pose[:3, 3] - current_pose[:3, 3]
    
    # Orientation error using matrix exponential map (axis-angle representation)
    # R_error = R_target @ R_current^T
    R_current = current_pose[:3, :3]
    R_target = target_pose[:3, :3]
    R_error = R_target @ R_current.T
    
    # Extract axis-angle representation from rotation matrix
    # Using the skew-symmetric part: (R - R^T) / 2 = [omega]_x
    skew = (R_error - R_error.T) / 2.0
    
    # Extract orientation error as vector from skew-symmetric matrix
    orientation_error = np.array([
        skew[2, 1],
        skew[0, 2],
        skew[1, 0]
    ])
    
    # Combine into 6D error vector
    error_6d = np.concatenate([position_error, orientation_error])
    
    return error_6d



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