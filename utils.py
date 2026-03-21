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