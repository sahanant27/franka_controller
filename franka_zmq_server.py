#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Copyright (c) 2025 Prabin Kumar Rath
# Co-developed with Claude Sonnet 4.6 (Anthropic)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

"""
Franka ZMQ Server — runs on Computer A (robot computer)

Starts a torque control loop via pylibfranka and exposes a ZMQ REP socket
so Computer B (GPU/camera) can read robot state and send targets.

Two control modes (switched dynamically by message type):
  - Joint impedance: tracks 7-DOF joint position targets
  - OSC (operational space): tracks end-effector pose targets

Usage:
    # Absolute EE targets (default):
    python franka_zmq_server.py --robot_ip 172.16.0.2 --port 5555

    # Delta EE targets (policy outputs are integrated as deltas on the server):
    python franka_zmq_server.py --robot_ip 172.16.0.2 --delta_ee

Message protocol (JSON over ZMQ REQ-REP):
    get_state      → {"q":[7 rad], "dq":[7 rad/s], "gripper":m,
                       "ee_pos":[x,y,z], "ee_rotvec":[rx,ry,rz], "timestamp":float}
    set_target     {"q":[7 rad]}              → {"status":"ok", "timestamp":float}
    set_ee_target  {"pos":[x,y,z], "rotvec":[rx,ry,rz]} → {"status":"ok", "timestamp":float}
                   With --delta_ee flag: values are integrated as deltas on the server.
    gripper_move   {"width":m, "speed":m/s}   → {"status":"ok", "timestamp":float}
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).parent))

from pylibfranka import RealtimeConfig, Robot, Torques

from utils.control import franka_array_to_matrix, limit_torque_rate
from utils.motion import goto_pose
from utils.transforms import axis_angle_to_rot_matrix, compute_pose_error, rot_matrix_to_axis_angle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("franka_zmq_server")


HOME_JOINTS = [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]

# Per-joint torque limits (Nm): Franka FR3 spec
MAX_TORQUES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Conservative collision thresholds (raise to 100 for policy deployment)
# TODO: Have to implement a data class for this
COLLISION_LOWER_TORQUES = [100.0] * 7
COLLISION_UPPER_TORQUES = [100.0] * 7
COLLISION_LOWER_FORCES = [100.0] * 6
COLLISION_UPPER_FORCES = [100.0] * 6


def pseudoinverse(matrix: np.ndarray, epsilon: float = 2.5e-4) -> np.ndarray:
    """SVD-based pseudoinverse with fixed singular-value cutoff (matches Deoxys)."""
    u, sv, vh = np.linalg.svd(matrix, full_matrices=True)
    sv_inv = np.zeros(matrix.shape, dtype=float)
    for i, s in enumerate(sv):
        if s >= epsilon:
            sv_inv[i, i] = 1.0 / s
    return vh.T @ sv_inv @ u.T


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Control mode — set by ZMQ thread, read by control loop
    control_mode: str = "joint"          # "joint" | "ee"
    # Joint space — written by ZMQ thread (joint mode) or read-only (ee mode)
    target_q: np.ndarray = field(default_factory=lambda: np.zeros(7))
    # EE space — written by ZMQ thread (ee mode)
    target_ee: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 pose
    # Written by control loop, read by ZMQ thread
    current_q: np.ndarray = field(default_factory=lambda: np.zeros(7))
    current_dq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    current_ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_ee_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_width: float = 0.08
    # Lifecycle
    control_running: bool = False
    shutdown: bool = False
    error: str | None = None


@dataclass
class ServerConfig:
    # Joint impedance gains
    stiffness: np.ndarray = field(default_factory=lambda: np.full(7, 200.0))
    damping: np.ndarray = field(default_factory=lambda: np.full(7, 28.28))  # 2*sqrt(200)
    # OSC gains: [tx, ty, tz, rx, ry, rz]
    ee_stiffness: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0, 30.0, 60.0, 60.0, 60.0]))
    ee_damping: np.ndarray = field(default_factory=lambda: np.array([10.95, 10.95, 10.95, 15.49, 15.49, 15.49]))  # 2*sqrt(gains)
    max_delta_tau: float = 1.0
    max_delta_per_msg: float = 0.1  # rad — per-ZMQ-message joint delta safety limit
    goto_home: bool = True
    # When True, set_ee_target values are treated as deltas and integrated on the server
    # using authoritative libfranka EE state. When False, values are absolute targets.
    delta_ee: bool = False


def _compute_ee_torque(
    robot_state,
    model,
    target_ee: np.ndarray,
    config: "ServerConfig",
) -> np.ndarray:
    """Compute OSC torques to track a 4x4 EE pose target.

    Uses block-diagonal operational-space inertia (separate lambda for
    translation and rotation) with a dynamically-consistent null-space
    projector — identical to OperationSpaceController._compute_torque().
    """
    dq = np.array(robot_state.dq)
    coriolis = np.array(model.coriolis(robot_state))
    mass_matrix = franka_array_to_matrix(model.mass(robot_state), (7, 7))
    jacobian = franka_array_to_matrix(model.zero_jacobian(robot_state), (6, 7))
    current_ee = franka_array_to_matrix(robot_state.O_T_EE, (4, 4))

    eef_velocity = jacobian @ dq
    error_6d = compute_pose_error(current_ee, target_ee)

    pos_error = error_6d[:3]
    ori_error = error_6d[3:]
    lin_vel = eef_velocity[:3]
    ang_vel = eef_velocity[3:]

    des_acc_pos = config.ee_stiffness[:3] * pos_error - config.ee_damping[:3] * lin_vel
    des_acc_ori = config.ee_stiffness[3:] * ori_error - config.ee_damping[3:] * ang_vel

    jacobian_pos = jacobian[:3, :]
    jacobian_ori = jacobian[3:, :]
    mass_inv = np.linalg.inv(mass_matrix)

    lambda_pos = pseudoinverse(jacobian_pos @ mass_inv @ jacobian_pos.T)
    lambda_ori = pseudoinverse(jacobian_ori @ mass_inv @ jacobian_ori.T)

    task_wrench_pos = lambda_pos @ des_acc_pos

    # Dynamically-consistent null-space projector for position task
    null_projector = np.eye(7) - jacobian_pos.T @ lambda_pos @ jacobian_pos @ mass_inv

    tau = (
        jacobian_pos.T @ task_wrench_pos
        + null_projector @ jacobian_ori.T @ (lambda_ori @ des_acc_ori)
        + coriolis
    )
    return tau


def control_loop_thread(robot_ip: str, shared: SharedState, config: ServerConfig) -> None:
    """Background thread: runs pylibfranka joint impedance torque control at ~1 kHz.

    The Robot object is owned entirely by this thread — the ZMQ thread must
    never call any robot.* methods.
    """
    try:
        logger.info(f"Connecting to robot at {robot_ip} ...")
        robot = Robot(robot_ip, RealtimeConfig.kIgnore)

        robot.set_collision_behavior(
            COLLISION_LOWER_TORQUES,
            COLLISION_UPPER_TORQUES,
            COLLISION_LOWER_FORCES,
            COLLISION_UPPER_FORCES,
        )

        if config.goto_home:
            logger.info("Moving to home position (goto_pose)...")
            # TODO: Make it a bit faster time constraint and all thats
            goto_pose(robot, target_joint_position=HOME_JOINTS)
            logger.info("Reached home position.")

        # Seed target from current state before starting active control
        initial_state = robot.read_once()
        initial_q = np.array(initial_state.q)

        with shared.lock:
            shared.target_q = initial_q.copy()
            shared.current_q = initial_q.copy()

        logger.info("Starting torque control loop...")
        active_control = robot.start_torque_control()
        model = robot.load_model()

        prev_tau_d = np.zeros(7)

        with shared.lock:
            shared.control_running = True

        while True:
            with shared.lock:
                if shared.shutdown:
                    break

            robot_state, _ = active_control.readOnce()

            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)
            coriolis = np.array(model.coriolis(robot_state))

            # Always extract EE pose so get_state can report it regardless of control mode
            current_ee = franka_array_to_matrix(robot_state.O_T_EE, (4, 4))
            ee_pos = current_ee[:3, 3]
            ee_rotvec = rot_matrix_to_axis_angle(current_ee[:3, :3])

            with shared.lock:
                mode = shared.control_mode
                q_target = shared.target_q.copy()
                ee_target = shared.target_ee.copy()
                shared.current_q = q.copy()
                shared.current_dq = dq.copy()
                shared.current_ee_pos = ee_pos.copy()
                shared.current_ee_rotvec = ee_rotvec.copy()

            if mode == "ee":
                tau_d = _compute_ee_torque(robot_state, model, ee_target, config)
            else:
                # Joint impedance: τ = -K*(q - q_target) - D*dq + coriolis
                tau_task = -config.stiffness * (q - q_target) - config.damping * dq
                tau_d = tau_task + coriolis

            # Torque rate limiting for smooth control (reuses utils/control.py)
            tau_d = limit_torque_rate(tau_d, prev_tau_d, config.max_delta_tau)
            tau_d = np.clip(tau_d, -MAX_TORQUES, MAX_TORQUES)
            prev_tau_d = tau_d.copy()

            torque_cmd = Torques(tau_d.tolist())
            torque_cmd.motion_finished = False
            active_control.writeOnce(torque_cmd)

        # Signal libfranka that motion is done before exiting the session
        torque_cmd = Torques(prev_tau_d.tolist())
        torque_cmd.motion_finished = True
        active_control.writeOnce(torque_cmd)
        logger.info("Control loop exited cleanly.")

    except Exception as e:
        logger.error(f"Control loop error: {e}", exc_info=True)
        with shared.lock:
            shared.shutdown = True
            shared.error = str(e)


def zmq_server_loop(shared: SharedState, port: int, config: ServerConfig) -> None:
    """Main thread: ZMQ REP socket handling get_state / set_target / gripper_move."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s poll interval for shutdown check
    sock.bind(f"tcp://*:{port}")
    logger.info(f"ZMQ REP server listening on tcp://*:{port}")

    try:
        while True:
            with shared.lock:
                if shared.shutdown:
                    break

            try:
                msg_bytes = sock.recv()
            except zmq.Again:
                # Timeout — loop back to check shutdown flag
                continue

            try:
                msg = json.loads(msg_bytes)
            except json.JSONDecodeError as e:
                sock.send_json({"status": "error", "message": f"JSON decode error: {e}"})
                continue

            msg_type = msg.get("type")

            if msg_type == "get_state":
                with shared.lock:
                    reply = {
                        "q": shared.current_q.tolist(),
                        "dq": shared.current_dq.tolist(),
                        "gripper": shared.gripper_width,
                        "ee_pos": shared.current_ee_pos.tolist(),
                        "ee_rotvec": shared.current_ee_rotvec.tolist(),
                        "timestamp": time.monotonic(),
                    }
                sock.send_json(reply)

            elif msg_type == "set_ee_target":
                try:
                    pos = np.array(msg["pos"], dtype=float)
                    rotvec = np.array(msg["rotvec"], dtype=float)
                    if len(pos) != 3 or len(rotvec) != 3:
                        sock.send_json({"status": "error", "message": "pos and rotvec must each have 3 elements"})
                        continue

                    target_ee = np.eye(4)

                    if config.delta_ee:
                        # Integrate delta on server using authoritative libfranka EE state
                        with shared.lock:
                            cur_pos = shared.current_ee_pos.copy()
                            cur_rotvec = shared.current_ee_rotvec.copy()
                        target_ee[:3, :3] = axis_angle_to_rot_matrix(rotvec) @ axis_angle_to_rot_matrix(cur_rotvec)
                        target_ee[:3, 3] = cur_pos + pos
                    else:
                        target_ee[:3, :3] = axis_angle_to_rot_matrix(rotvec)
                        target_ee[:3, 3] = pos

                    with shared.lock:
                        shared.target_ee = target_ee
                        shared.control_mode = "ee"

                    sock.send_json({"status": "ok", "timestamp": time.monotonic()})

                except (KeyError, ValueError) as e:
                    sock.send_json({"status": "error", "message": str(e)})

            elif msg_type == "set_target":
                try:
                    new_q = np.array(msg["q"], dtype=float)
                    if len(new_q) != 7:
                        sock.send_json({"status": "error", "message": "q must have 7 elements"})
                        continue

                    with shared.lock:
                        current_q = shared.current_q.copy()
                        delta = np.clip(
                            new_q - current_q,
                            -config.max_delta_per_msg,
                            config.max_delta_per_msg,
                        )
                        shared.target_q = current_q + delta
                        shared.control_mode = "joint"

                    sock.send_json({"status": "ok", "timestamp": time.monotonic()})

                except (KeyError, ValueError) as e:
                    sock.send_json({"status": "error", "message": str(e)})

            elif msg_type == "gripper_move":
                width = float(msg.get("width", 0.08))
                speed = float(msg.get("speed", 0.1))

                # TODO: Replace this stub with actual pylibfranka Gripper control once
                # the gripper API is confirmed. Example:
                #   from pylibfranka import Gripper
                #   gripper = Gripper(robot_ip)   # created once in main()
                #   threading.Thread(target=gripper.move, args=(width, speed), daemon=True).start()
                with shared.lock:
                    shared.gripper_width = width
                logger.info(
                    f"Gripper move: width={width:.4f}m speed={speed:.3f}m/s "
                    "(stub — implement Gripper.move() call here)"
                )
                sock.send_json({"status": "ok", "timestamp": time.monotonic()})

            else:
                sock.send_json({"status": "error", "message": f"Unknown type: {msg_type!r}"})

    finally:
        sock.close()
        ctx.term()
        logger.info("ZMQ server closed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Franka ZMQ Server: joint impedance control with ZMQ command interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot_ip", default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ REP port")
    parser.add_argument(
        "--stiffness", type=float, nargs=7,
        default=[200.0] * 7,
        metavar=("K1", "K2", "K3", "K4", "K5", "K6", "K7"),
        help="Joint stiffness gains (Nm/rad). Higher = stiffer tracking.",
    )
    parser.add_argument(
        "--damping", type=float, nargs=7,
        default=None,
        metavar=("D1", "D2", "D3", "D4", "D5", "D6", "D7"),
        help="Joint damping gains (Nm·s/rad). Default: 2*sqrt(stiffness) (critical damping).",
    )
    parser.add_argument(
        "--max_delta_tau", type=float, default=1.0,
        help="Max torque change per 1kHz control step (Nm). Lower = smoother but slower response.",
    )
    parser.add_argument(
        "--max_delta_per_msg", type=float, default=0.1,
        help="Max joint delta per ZMQ set_target message (rad). Safety coarse-clipping.",
    )
    parser.add_argument(
        "--no_goto_home", action="store_true",
        help="Skip homing motion at startup (useful for debugging).",
    )
    parser.add_argument(
        "--delta_ee", action="store_true",
        help=(
            "Treat set_ee_target values as deltas and integrate on the server "
            "using authoritative libfranka EE state. "
            "When not set (default), set_ee_target values are absolute pose targets."
        ),
    )
    args = parser.parse_args()

    stiffness = np.array(args.stiffness)
    damping = 2.0 * np.sqrt(stiffness) if args.damping is None else np.array(args.damping)

    config = ServerConfig(
        stiffness=stiffness,
        damping=damping,
        max_delta_tau=args.max_delta_tau,
        max_delta_per_msg=args.max_delta_per_msg,
        goto_home=not args.no_goto_home,
        delta_ee=args.delta_ee,
    )

    shared = SharedState()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received, stopping...")
        with shared.lock:
            shared.shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    control_thread = threading.Thread(
        target=control_loop_thread,
        args=(args.robot_ip, shared, config),
        daemon=True,
        name="franka_control",
    )
    control_thread.start()

    # Wait for control loop to initialize (goto_pose can take ~10s)
    startup_timeout = 30.0
    start = time.time()
    while True:
        with shared.lock:
            if shared.control_running:
                break
            if shared.shutdown or shared.error:
                logger.error(f"Control loop failed to start: {shared.error}")
                return 1
        if time.time() - start > startup_timeout:
            logger.error("Timed out waiting for control loop to start")
            return 1
        time.sleep(0.1)

    logger.info("Control loop running. ZMQ server starting...")
    zmq_server_loop(shared, args.port, config)

    control_thread.join(timeout=5.0)
    if control_thread.is_alive():
        logger.warning("Control thread did not exit within timeout")

    logger.info("Server shut down cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())





