import argparse
import time

import numpy as np

from pylibfranka import  RealtimeConfig, Robot, Torques

from utils import (
    compute_6d_error,
    create_delta_frame_rotation_axis,
    create_delta_frame_translation_axis,
    goto_pose,
)


def franka_array_to_matrix(values, shape):
    """Convert a Franka column-major array to a matrix with the requested shape."""
    return np.array(values).reshape(*shape, order="F")


def saturate_torque_rate(tau_d_calculated, tau_j_d, max_delta_tau):
    """Limit per-joint torque step to keep commands continuous for libfranka."""
    delta_tau = tau_d_calculated - tau_j_d
    delta_tau = np.clip(delta_tau, -max_delta_tau, max_delta_tau)
    return tau_j_d + delta_tau


def compute_error_norms(error_6d):
    """Return translation, rotation, and combined norms from a 6D OSC error."""
    trans_norm = np.linalg.norm(error_6d[:3])
    rot_norm = np.linalg.norm(error_6d[3:])
    total_norm = np.linalg.norm(error_6d)
    return trans_norm, rot_norm, total_norm


def print_error_norms(label, norms, time_elapsed=None):
    """Print translation, rotation, and combined norms from a precomputed tuple."""
    trans_norm, rot_norm, total_norm = norms
    message = (
        f"{label}: total={total_norm:.6f}, "
        f"translation={trans_norm:.6f} m, rotation={rot_norm:.6f} rad"
    )
    if time_elapsed is not None:
        message = f"t={time_elapsed:.2f}s | {message}"
    print(message)

def rot_matrix_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle

def axis_angle_to_rot_matrix(rvec):
    angle = np.linalg.norm(rvec)
    if angle < 1e-6:
        return np.eye(3)
    axis = rvec / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def slerp_rot_matrix(R1, R2, t):
    R_rel = R1.T @ R2
    rvec = rot_matrix_to_axis_angle(R_rel)
    return R1 @ axis_angle_to_rot_matrix(rvec * t)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip, RealtimeConfig.kIgnore)

    decay = 0.995

    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # First move the robot to a suitable joint configuration
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")
        print("Moving to home configuration...")
        goto_pose(robot)
        print("Home reached. Starting operation-space control.")
        active_control = robot.start_torque_control()

        time_elapsed = 0.0
        motion_finished = False
        # NOTE: set default values for now
        
        max_torques = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # Conservative torque limits for testing

        # Get initial state and model
        robot_state, duration = active_control.readOnce()
        initial_cartesian_pose = franka_array_to_matrix(robot_state.O_T_EE, (4, 4))

        model = robot.load_model()

        # Build target pose from single-axis delta transforms for controller checks.
        # Keep only ONE test active at a time.
        # delta_translation = create_delta_frame_translation_axis("x", 0.10)  # Active: +10 cm on X
        # delta_translation = create_delta_frame_translation_axis("y", 0.10)  # +10 cm on Y
        # delta_translation = create_delta_frame_translation_axis("z", 0.10)  # +10 cm on Z
        delta_translation = np.eye(4)  # Active: no translation change
        
        delta_rotation = np.eye(4)  # Active: no rotation change
        delta_rotation = create_delta_frame_rotation_axis("x", np.deg2rad(5.0))  # +5 deg about X
        # delta_rotation = create_delta_frame_rotation_axis("y", np.deg2rad(5.0))  # +5 deg about Y
        # delta_rotation = create_delta_frame_rotation_axis("z", np.deg2rad(5.0))  # +5 deg about Z

        # Apply translation first, then rotation in the EE/local frame.
        target_frame = initial_cartesian_pose @ delta_translation @ delta_rotation
        
        # Motion and damping gains
        base_gains = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])
        motion_gains = base_gains.copy()
        damping_gains = 2.0 * np.sqrt(base_gains)

        # Error threshold for motion completion
        error_threshold = 1e-3
        
        # Trajectory duration
        trajectory_duration = 5.0        
        # Torque clamping for conservative testing
        max_delta_tau = 1.0
        next_error_print_time = 0.0

        # Initialize current state values for EMA smoothing
        current_motion_gains = motion_gains.copy()
        current_damping_gains = damping_gains.copy()
        current_target_frame = initial_cartesian_pose.copy()

        initial_error_6d = compute_6d_error(initial_cartesian_pose, target_frame)
        print_error_norms("initial error", compute_error_norms(initial_error_6d))

        # External control loop
        while not motion_finished:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            M = franka_array_to_matrix(model.mass(robot_state), (7, 7))
            J = franka_array_to_matrix(model.zero_jacobian(robot_state), (6, 7))
            # Update time
            time_elapsed += duration.to_sec()
            
            # EMA Smoothing
            current_motion_gains = decay * current_motion_gains + (1 - decay) * motion_gains
            current_damping_gains = decay * current_damping_gains + (1 - decay) * damping_gains
            
            # Position Lerp
            current_target_frame[0:3, 3] = decay * current_target_frame[0:3, 3] + (1 - decay) * target_frame[0:3, 3]
            
            # Rotation Slerp
            R_current = current_target_frame[0:3, 0:3]
            R_target = target_frame[0:3, 0:3]
            current_target_frame[0:3, 0:3] = slerp_rot_matrix(R_current, R_target, 1 - decay)

            # Get current pose
            current_pose = franka_array_to_matrix(robot_state.O_T_EE, (4, 4))

            coriolis = np.array(model.coriolis(robot_state))

            dq = np.array(robot_state.dq)
            eef_velocity = J @ dq
            # Compute 6D error for OSC

            error_6d = compute_6d_error(current_pose, current_target_frame)

            des_acc = current_motion_gains * error_6d - current_damping_gains * eef_velocity

            I = np.eye(6)
            M_inv = np.linalg.inv(M)
            OSM = np.linalg.inv(J @ M_inv @ J.T + 1e-3 * I)  
            tau_d = J.T @ OSM @ des_acc + coriolis

            # Enforce torque continuity relative to robot controller's desired torque.
            tau_j_d = np.array(robot_state.tau_J_d)
            tau_d = saturate_torque_rate(tau_d, tau_j_d, max_delta_tau)

            # NOTE: Add nullspace later.
            tau_d = np.clip(tau_d, -max_torques, max_torques)
            # print("tau_d:", tau_d)
            torque_command = Torques(tau_d.tolist())
            torque_command.motion_finished = False

            # Trajectory completion check - if error is close to zero
            final_error_6d = compute_6d_error(current_pose, target_frame)
            final_norms = compute_error_norms(final_error_6d)
            final_trans_norm, final_rot_norm, final_total_norm = final_norms
            if final_total_norm < error_threshold:
                torque_command.motion_finished = True
                motion_finished = True
                print("Trajectory finished (error below threshold), shutting down example")
                print_error_norms("Final errors", final_norms)
            elif time_elapsed >= trajectory_duration:
                # Fallback: finish if time expires
                torque_command.motion_finished = True
                motion_finished = True
                print("Trajectory finished (time limit reached), shutting down example")
                print_error_norms("Final errors", final_norms)

            if time_elapsed >= next_error_print_time:
                tracked_error_6d = compute_6d_error(current_pose, current_target_frame)
                tracked_norms = compute_error_norms(tracked_error_6d)
                print_error_norms("final_target", final_norms, time_elapsed=time_elapsed)
                print_error_norms("tracked_target", tracked_norms, time_elapsed=time_elapsed)
                next_error_print_time += 1.0
            


            # Send command to robot
            active_control.writeOnce(torque_command)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
