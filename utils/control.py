import numpy as np


def franka_array_to_matrix(values, shape):
    """Convert Franka column-major array data to a matrix."""
    return np.array(values).reshape(*shape, order="F")


def limit_torque_rate(tau_desired, tau_reference, max_delta_tau):
    """Limit per-joint torque step for torque continuity."""
    delta_tau = tau_desired - tau_reference
    delta_tau = np.clip(delta_tau, -max_delta_tau, max_delta_tau)
    return tau_reference + delta_tau


def pose_error_norms(error_6d):
    """Return translation, rotation and combined norms from a 6D error."""
    trans_norm = np.linalg.norm(error_6d[:3])
    rot_norm = np.linalg.norm(error_6d[3:])
    total_norm = np.linalg.norm(error_6d)
    return trans_norm, rot_norm, total_norm


def log_pose_error(label, norms, time_elapsed=None, include_total=True):
    """Print a formatted pose-error line."""
    trans_norm, rot_norm, total_norm = norms
    if include_total:
        message = (
            f"{label}: total={total_norm:.6f}, "
            f"translation={trans_norm:.6f} m, rotation={rot_norm:.6f} rad"
        )
    else:
        message = f"{label}: translation={trans_norm:.6f} m, rotation={rot_norm:.6f} rad"
    if time_elapsed is not None:
        message = f"t={time_elapsed:.2f}s | {message}"
    print(message)
