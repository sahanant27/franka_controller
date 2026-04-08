import numpy as np


# def matrix_to_rpy(rotation_matrix):
#     """Convert a rotation matrix to roll, pitch, yaw (radians)."""
#     if rotation_matrix.shape == (4, 4):
#         R = rotation_matrix[:3, :3]
#     else:
#         R = rotation_matrix

#     sin_pitch = np.clip(-R[2, 0], -1.0, 1.0)
#     pitch = np.arcsin(sin_pitch)

#     cos_pitch = np.cos(pitch)
#     if np.abs(cos_pitch) > 1e-6:
#         roll = np.arctan2(R[2, 1], R[2, 2])
#         yaw = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         roll = 0.0
#         yaw = np.arctan2(-R[0, 1], R[1, 1])

#     return roll, pitch, yaw


# def matrix_to_quaternion(transformation_matrix):
#     """Convert a 4x4 transform matrix to quaternion in (w, x, y, z)."""
#     R = transformation_matrix[:3, :3]
#     trace = R[0, 0] + R[1, 1] + R[2, 2]

#     if trace > 0:
#         s = 0.5 / np.sqrt(trace + 1.0)
#         w = 0.25 / s
#         x = (R[2, 1] - R[1, 2]) * s
#         y = (R[0, 2] - R[2, 0]) * s
#         z = (R[1, 0] - R[0, 1]) * s
#     elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
#         s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
#         w = (R[2, 1] - R[1, 2]) / s
#         x = 0.25 * s
#         y = (R[0, 1] + R[1, 0]) / s
#         z = (R[0, 2] + R[2, 0]) / s
#     elif R[1, 1] > R[2, 2]:
#         s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
#         w = (R[0, 2] - R[2, 0]) / s
#         x = (R[0, 1] + R[1, 0]) / s
#         y = 0.25 * s
#         z = (R[1, 2] + R[2, 1]) / s
#     else:
#         s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
#         w = (R[1, 0] - R[0, 1]) / s
#         x = (R[0, 2] + R[2, 0]) / s
#         y = (R[1, 2] + R[2, 1]) / s
#         z = 0.25 * s

#     return np.array([w, x, y, z])


# def create_target_frame_translation_only(position):
#     """Create a target frame with identity rotation and given translation."""
#     T = np.eye(4)
#     T[0, 3] = position[0]
#     T[1, 3] = position[1]
#     T[2, 3] = position[2]
#     return T


# def _axis_to_index(axis):
#     """Map axis specifier ('x'/'y'/'z' or 0/1/2) to an integer index."""
#     if isinstance(axis, str):
#         axis = axis.lower()
#         axis_map = {"x": 0, "y": 1, "z": 2}
#         if axis not in axis_map:
#             raise ValueError("axis must be one of: 'x', 'y', 'z', 0, 1, 2")
#         return axis_map[axis]

#     if axis in (0, 1, 2):
#         return axis

#     raise ValueError("axis must be one of: 'x', 'y', 'z', 0, 1, 2")


# def create_delta_frame_translation_axis(axis, distance):
#     """Create a 4x4 delta transform with translation only along one axis."""
#     axis_idx = _axis_to_index(axis)
#     T_delta = np.eye(4)
#     T_delta[axis_idx, 3] = distance
#     return T_delta


# def create_delta_frame_rotation_axis(axis, angle_rad):
#     """Create a 4x4 delta transform with rotation only about one axis."""
#     axis_idx = _axis_to_index(axis)
#     c = np.cos(angle_rad)
#     s = np.sin(angle_rad)

#     if axis_idx == 0:
#         R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
#     elif axis_idx == 1:
#         R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
#     else:
#         R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

#     T_delta = np.eye(4)
#     T_delta[:3, :3] = R
#     return T_delta


# def rot_matrix_to_axis_angle(R):
#     """Convert a rotation matrix to axis-angle vector (axis * angle)."""
#     angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
#     if angle < 1e-6:
#         return np.zeros(3)
#     axis = np.array(
#         [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
#     ) / (2 * np.sin(angle))
#     return axis * angle


# def axis_angle_to_rot_matrix(rvec):
#     """Convert axis-angle vector (axis * angle) to a rotation matrix."""
#     angle = np.linalg.norm(rvec)
#     if angle < 1e-6:
#         return np.eye(3)

#     axis = rvec / angle
#     K = np.array(
#         [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
#     )
#     return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# def slerp_rot_matrix(R1, R2, t):
#     """Interpolate between two rotation matrices using axis-angle slerp."""
#     R_rel = R1.T @ R2
#     rvec = rot_matrix_to_axis_angle(R_rel)
#     return R1 @ axis_angle_to_rot_matrix(rvec * t)


# def quat_multiply(q1, q2):
#     """Multiply two quaternions in (w, x, y, z) format."""
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
    
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
#     return np.array([w, x, y, z])


# def quat_conjugate(q):
#     """Return conjugate of quaternion in (w, x, y, z) format."""
#     return np.array([q[0], -q[1], -q[2], -q[3]])


# def quat_to_axis_angle(q):
#     """Convert quaternion (w, x, y, z) to axis-angle vector."""
#     w, x, y, z = q
#     # Normalize quaternion
#     norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
#     w, x, y, z = w / norm, x / norm, y / norm, z / norm
    
#     # Clamp w to avoid numerical issues with arccos
#     w = np.clip(w, -1.0, 1.0)
    
#     # Compute angle
#     angle = 2.0 * np.arccos(w)
    
#     # Compute axis
#     sin_half_angle = np.sqrt(1.0 - w**2)
#     if sin_half_angle < 1e-6:
#         # Small angle: axis is undefined, return zero vector
#         return np.zeros(3)
    
#     axis = np.array([x, y, z]) / sin_half_angle
#     return axis * angle


# def compute_pose_error(current_pose, target_pose):
#     """Compute position and orientation error using quaternion-based approach.
    
#     Returns error as target minus current for use with +K*error - D*x_dot.
#     Position error is (3,), orientation error is (3,) as axis-angle vector.
#     """
#     # Compute position error
#     position_error = target_pose[:3, 3] - current_pose[:3, 3]
    
#     # Convert rotation matrices to quaternions
#     q_current = matrix_to_quaternion(current_pose)
#     q_target = matrix_to_quaternion(target_pose)
    
#     # Compute quaternion error: q_error = q_target * q_current^{-1}
#     q_current_conj = quat_conjugate(q_current)
#     q_error = quat_multiply(q_target, q_current_conj)
    
#     # Convert quaternion error to axis-angle
#     orientation_error = quat_to_axis_angle(q_error)
    
#     return np.concatenate([position_error, orientation_error])
