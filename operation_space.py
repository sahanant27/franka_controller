import argparse
import time

import numpy as np

from pylibfranka import  RealtimeConfig, Robot, Torques

from utils import compute_6d_error, create_target_frame_translation_only


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip, RealtimeConfig.kIgnore)

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
        max_torques = np.array([20.0, 20.0, 20.0, 20.0, 6.0, 6.0, 6.0])  # Conservative torque limits for testing

        # Get initial state and model
        robot_state, duration = active_control.readOnce()
        initial_cartesian_pose = np.array(robot_state.O_T_EE).reshape(4, 4)

        model = robot.load_model()
        # Define target position (translation only, no rotation change)
        target_position = np.array([
            initial_cartesian_pose[0, 3] + 0.1,  # x offset
            initial_cartesian_pose[1, 3],         # y (unchanged)
            initial_cartesian_pose[2, 3]          # z (unchanged)
        ])

        target_frame = initial_cartesian_pose.copy()
        target_frame[0:3, 3] = target_position
        
        # Motion and damping gains
        base_gains = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])
        motion_gains = base_gains.copy()
        damping_gains = 2.0 * np.sqrt(base_gains)

        # Error threshold for motion completion
        error_threshold = 1e-3
        
        # Trajectory duration
        trajectory_duration = 5.0        
        # Torque clamping for conservative testing
        max_delta_tau = 0.1
        prev_tau_d = np.zeros(7)         
        # External control loop
        while not motion_finished:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            M = np.array(model.mass(robot_state)).reshape(7, 7)
            J = np.array(model.zero_jacobian(robot_state)).reshape(6, 7)
            # Update time
            time_elapsed += duration.to_sec()
            
            # Smooth trajectory generation (minimum jerk)
            tau = min(time_elapsed / trajectory_duration, 1.0)
            alpha = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            
            current_target_frame = initial_cartesian_pose.copy()
            current_target_frame[0:3, 3] = initial_cartesian_pose[0:3, 3] + alpha * (target_position - initial_cartesian_pose[0:3, 3])

            # Get current pose
            current_pose = np.array(robot_state.O_T_EE).reshape(4, 4)

            coriolis = np.array(model.coriolis(robot_state))

            dq = np.array(robot_state.dq)
            eef_velocity = J @ dq
            # Compute 6D error for OSC

            error_6d = compute_6d_error(current_pose, current_target_frame)

            des_acc = motion_gains * error_6d - damping_gains * eef_velocity

            I = np.eye(6)
            M_inv = np.linalg.inv(M)
            OSM = np.linalg.inv(J @ M_inv @ J.T + 1e-3 * I)  
            tau_d = J.T @ OSM @ des_acc + coriolis

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
