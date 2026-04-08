import numpy as np


def _as_rotation_matrix(matrix):
    """Return the 3x3 rotation block from a 3x3 or 4x4 matrix."""
    return matrix[:3, :3] if matrix.shape == (4, 4) else matrix


def _matrix_to_quaternion(matrix):
    """Convert a 3x3 or 4x4 rotation matrix to quaternion [w, x, y, z]."""
    rotation_matrix = _as_rotation_matrix(matrix)
    trace = np.trace(rotation_matrix)

    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * scale
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * scale
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * scale
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def _quat_conjugate(quaternion):
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=float)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def matrix_to_rpy(rotation_matrix):
    """Convert a rotation matrix to roll, pitch, yaw (radians)."""
    rotation_matrix = _as_rotation_matrix(rotation_matrix)

    sin_pitch = np.clip(-rotation_matrix[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)

    if np.abs(cos_pitch) > 1e-6:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

    return roll, pitch, yaw


def create_frame_from_xyzrpy(
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    base_frame=None,
):
    """Create a pose from xyz/rpy, optionally composed with a given base frame."""
    x, y, z = np.asarray(xyz, dtype=float)
    roll, pitch, yaw = np.asarray(rpy, dtype=float)

    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    rotation = np.array(
        [
            [
                cos_yaw * cos_pitch,
                cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
            ],
            [
                sin_yaw * cos_pitch,
                sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
            ],
            [
                -sin_pitch,
                cos_pitch * sin_roll,
                cos_pitch * cos_roll,
            ],
        ]
    )

    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = np.array([x, y, z], dtype=float)

    if base_frame is not None:
        return np.asarray(base_frame, dtype=float) @ target
    return target


def rot_matrix_to_axis_angle(rotation_matrix):
    rotation_matrix = _as_rotation_matrix(rotation_matrix)
    angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def axis_angle_to_rot_matrix(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-6:
        return np.eye(3)

    axis = rotation_vector / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def slerp_rot_matrix(start_rotation, goal_rotation, fraction):
    start_rotation = _as_rotation_matrix(start_rotation)
    goal_rotation = _as_rotation_matrix(goal_rotation)
    relative_rotation = start_rotation.T @ goal_rotation
    rotation_vector = rot_matrix_to_axis_angle(relative_rotation)
    return start_rotation @ axis_angle_to_rot_matrix(rotation_vector * fraction)


def compute_pose_error(current_pose, target_pose):
    """Compute 6D pose error using a Franka-style orientation convention."""
    position_error = target_pose[:3, 3] - current_pose[:3, 3]
    current_rotation = _as_rotation_matrix(current_pose)
    q_current = _matrix_to_quaternion(current_rotation)
    q_target = _matrix_to_quaternion(target_pose)

    if np.dot(q_target, q_current) < 0.0:
        q_current = -q_current

    error_quaternion = _quat_multiply(_quat_conjugate(q_current), q_target)
    orientation_error = -current_rotation @ error_quaternion[1:]
    return np.concatenate([position_error, orientation_error])
