import argparse
import time

import numpy as np

from pylibfranka import  RealtimeConfig, Robot, Torques

from utils import compute_6d_error, create_target_frame_translation_only

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

# EMA Filter
# torque rate limiter
# nullspace control
# Integral error compensation 


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
        active_control = robot.start_torque_control()

        time_elapsed = 0.0
        motion_finished = False
        max_torques = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # Conservative torque limits for testing


        # Get initial state and model
        robot_state, duration = active_control.readOnce()
        initial_cartesian_pose = np.array(robot_state.O_T_EE).reshape(4, 4)

        model = robot.load_model()
        target_position = np.array([
            initial_cartesian_pose[0, 3] + 0.1,  # x offset
            initial_cartesian_pose[1, 3],         # y (unchanged)
            initial_cartesian_pose[2, 3]          # z (unchanged)
        ])

        target_frame = initial_cartesian_pose.copy()
        target_frame[0:3, 3] = target_position
        
        # Motion and damping gains
        base_gains = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])
        ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Integral gains set to zero for now
        motion_gains = base_gains.copy()
        damping_gains = 2.0 * np.sqrt(base_gains)

        # Error threshold for motion completion
        error_threshold = 1e-3
        
        # Trajectory duration
        trajectory_duration = 5.0        
        # Torque clamping for conservative testing
        max_delta_tau = 1.0
        prev_tau_d = np.zeros(7)   

        error_i = np.zeros(6)
        rot_threshold = 0.3
        trans_threshold =0.1

        # Initialize current state values for EMA smoothing
        current_motion_gains = motion_gains.copy()
        current_damping_gains = damping_gains.copy()
        current_target_frame = initial_cartesian_pose.copy()

        # External control loop
        while not motion_finished:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            J = np.array(model.zero_jacobian(robot_state)).reshape(6, 7)
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
            current_pose = np.array(robot_state.O_T_EE).reshape(4, 4)

            coriolis = np.array(model.coriolis(robot_state))

            dq = np.array(robot_state.dq)
            eef_velocity = J @ dq
            # Compute 6D error for OSC

            error_6d = compute_6d_error(current_pose, current_target_frame)
            # Clamp first 3 and the other 3 separately

            error_i += np.concatenate([
                error_6d[..., :3].clip(min=-trans_threshold, max=trans_threshold),
                error_6d[..., 3:].clip(min=-rot_threshold, max=rot_threshold)
            ], axis=-1)

            des_acc = current_motion_gains * error_6d - current_damping_gains * eef_velocity + ki * error_i

            # I = np.eye(6)
            # M_inv = np.linalg.inv(M)
            # OSM = np.linalg.inv(J @ M_inv @ J.T + 1e-3 * I)
            # CI = None # Placeholder for Composite Inertia, can be replaced with actual computation if needed 
            tau_d = J.T  @ des_acc + coriolis

            # Apply torque clamping for conservative testing
            delta_tau = tau_d - prev_tau_d
            delta_tau = np.clip(delta_tau, -max_delta_tau, max_delta_tau)
            tau_d = prev_tau_d + delta_tau

            # NOTE: Add nullspace later.
            tau_d = np.clip(tau_d, -max_torques, max_torques)
            prev_tau_d = tau_d.copy()
            # print("tau_d:", tau_d)
            torque_command = Torques(tau_d.tolist())
            torque_command.motion_finished = False

            # Trajectory completion check - if error is close to zero
            final_error_6d = compute_6d_error(current_pose, target_frame)
            error_magnitude = np.linalg.norm(final_error_6d)
            if error_magnitude < error_threshold:
                torque_command.motion_finished = True
                motion_finished = True
                print("Trajectory finished (error below threshold), shutting down example")
                print(f"Final 6D error magnitude: {error_magnitude}")
            elif time_elapsed >= trajectory_duration:
                # Fallback: finish if time expires
                torque_command.motion_finished = True
                motion_finished = True
                print("Trajectory finished (time limit reached), shutting down example")
                print(f"Final 6D error magnitude: {error_magnitude}")

            # print(f"6D error magnitude: {error_magnitude}")
            


            # Send command to robot
            active_control.writeOnce(torque_command)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
