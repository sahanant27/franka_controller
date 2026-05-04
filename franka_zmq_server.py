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

Owns the libfranka connection. Holds the robot at the last commanded joint
position using a minimal joint impedance loop, and exposes a ZMQ REP socket
so run_policy.py can read robot state and send joint-position targets.

All controller math (OSC, IK, etc.) lives in run_policy.py — this file is
a pure state-relay + joint-hold loop.

Usage:
    python franka_zmq_server.py --robot_ip 172.16.0.2 --port 5555

Message protocol (JSON over ZMQ REQ-REP):
    get_state   → {"q":[7], "dq":[7], "ee_pos":[3], "ee_rotvec":[3],
                   "gripper":float, "timestamp":float}
    set_target  {"q":[7]}  → {"status":"ok", "timestamp":float}
    gripper_move{"width":float, "speed":float} → {"status":"ok", "timestamp":float}
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import zmq

from pylibfranka import RealtimeConfig, Robot, Torques


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("franka_zmq_server")


# ---------------------------------------------------------------------------
# Coordinate helpers (state reporting only — not controller math)
# ---------------------------------------------------------------------------

def _col_major_to_matrix(values, shape: tuple) -> np.ndarray:
    return np.array(values).reshape(*shape, order="F")


def _rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → axis-angle vector."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


# ---------------------------------------------------------------------------
# Startup homing (runs once before ZMQ loop opens)
# ---------------------------------------------------------------------------

def _goto_home(robot, target_q: list, duration: float = 3.0) -> None:
    """Drive robot to target_q with joint impedance torque control, then exit."""
    target_q = np.array(target_q)
    stiffness = np.array([50.0] * 7)
    damping = 2.0 * np.sqrt(stiffness)

    initial_state = robot.read_once()
    start_q = np.array(initial_state.q)

    active_control = robot.start_torque_control()
    model = robot.load_model()

    t0 = time.time()
    converged_since = None

    while True:
        state, _ = active_control.readOnce()
        q = np.array(state.q)
        dq = np.array(state.dq)
        coriolis = np.array(model.coriolis(state))

        # Minimum-jerk interpolated goal
        s = min((time.time() - t0) / duration, 1.0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        q_goal = start_q + s * (target_q - start_q)

        tau = -stiffness * (q - q_goal) - damping * dq + coriolis
        cmd = Torques(tau.tolist())
        cmd.motion_finished = False
        active_control.writeOnce(cmd)

        pos_ok = np.all(np.abs(target_q - q) <= 1e-3)
        vel_ok = np.all(np.abs(dq) <= 5e-3)
        now = time.time()
        if pos_ok and vel_ok:
            converged_since = converged_since or now
            if now - converged_since >= 0.2:
                cmd.motion_finished = True
                active_control.writeOnce(cmd)
                return
        else:
            converged_since = None

        if now - t0 >= duration + 5.0:
            cmd.motion_finished = True
            active_control.writeOnce(cmd)
            return


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_JOINTS = [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]

MAX_TORQUES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

COLLISION_LOWER_TORQUES = [100.0] * 7
COLLISION_UPPER_TORQUES = [100.0] * 7
COLLISION_LOWER_FORCES  = [100.0] * 6
COLLISION_UPPER_FORCES  = [100.0] * 6


# ---------------------------------------------------------------------------
# Server state / config
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    target_q: np.ndarray = field(default_factory=lambda: np.zeros(7))
    current_q: np.ndarray = field(default_factory=lambda: np.zeros(7))
    current_dq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    current_ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_ee_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_width: float = 0.08
    control_running: bool = False
    shutdown: bool = False
    error: str | None = None


@dataclass
class ServerConfig:
    stiffness: np.ndarray = field(default_factory=lambda: np.full(7, 200.0))
    damping: np.ndarray = field(default_factory=lambda: np.full(7, 28.28))
    max_delta_tau: float = 1.0
    max_delta_per_msg: float = 0.1   # rad — per-message joint delta safety clip
    goto_home: bool = True


# ---------------------------------------------------------------------------
# Control loop — joint impedance hold only, no controller math
# ---------------------------------------------------------------------------

def control_loop_thread(robot_ip: str, shared: SharedState, config: ServerConfig) -> None:
    """1 kHz joint-impedance hold loop. Owned entirely by this thread."""
    try:
        logger.info(f"Connecting to robot at {robot_ip} ...")
        robot = Robot(robot_ip, RealtimeConfig.kIgnore)
        robot.set_collision_behavior(
            COLLISION_LOWER_TORQUES, COLLISION_UPPER_TORQUES,
            COLLISION_LOWER_FORCES,  COLLISION_UPPER_FORCES,
        )

        if config.goto_home:
            logger.info("Moving to home position...")
            _goto_home(robot, HOME_JOINTS)
            logger.info("Reached home position.")

        initial_state = robot.read_once()
        initial_q = np.array(initial_state.q)
        with shared.lock:
            shared.target_q = initial_q.copy()
            shared.current_q = initial_q.copy()

        logger.info("Starting joint-impedance hold loop...")
        active_control = robot.start_torque_control()
        model = robot.load_model()
        prev_tau = np.zeros(7)

        with shared.lock:
            shared.control_running = True

        while True:
            with shared.lock:
                if shared.shutdown:
                    break
                q_target = shared.target_q.copy()

            state, _ = active_control.readOnce()
            q = np.array(state.q)
            dq = np.array(state.dq)
            coriolis = np.array(model.coriolis(state))

            # Update shared state for ZMQ reads
            ee = _col_major_to_matrix(state.O_T_EE, (4, 4))
            with shared.lock:
                shared.current_q = q.copy()
                shared.current_dq = dq.copy()
                shared.current_ee_pos = ee[:3, 3].copy()
                shared.current_ee_rotvec = _rot_to_rotvec(ee[:3, :3]).copy()

            # Joint impedance: τ = −K·(q − q_d) − D·dq + coriolis
            tau = -config.stiffness * (q - q_target) - config.damping * dq + coriolis

            # Safety: rate-limit and clip
            tau = prev_tau + np.clip(tau - prev_tau, -config.max_delta_tau, config.max_delta_tau)
            tau = np.clip(tau, -MAX_TORQUES, MAX_TORQUES)
            prev_tau = tau.copy()

            cmd = Torques(tau.tolist())
            cmd.motion_finished = False
            active_control.writeOnce(cmd)

        cmd = Torques(prev_tau.tolist())
        cmd.motion_finished = True
        active_control.writeOnce(cmd)
        logger.info("Control loop exited cleanly.")

    except Exception as e:
        logger.error(f"Control loop error: {e}", exc_info=True)
        with shared.lock:
            shared.shutdown = True
            shared.error = str(e)


# ---------------------------------------------------------------------------
# ZMQ relay
# ---------------------------------------------------------------------------

def zmq_server_loop(shared: SharedState, port: int, config: ServerConfig) -> None:
    """Main thread: REP socket — get_state / set_target / gripper_move."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://*:{port}")
    logger.info(f"ZMQ REP listening on tcp://*:{port}")

    try:
        while True:
            with shared.lock:
                if shared.shutdown:
                    break

            try:
                msg_bytes = sock.recv()
            except zmq.Again:
                continue

            try:
                msg = json.loads(msg_bytes)
            except json.JSONDecodeError as e:
                sock.send_json({"status": "error", "message": f"JSON decode: {e}"})
                continue

            msg_type = msg.get("type")

            if msg_type == "get_state":
                with shared.lock:
                    reply = {
                        "q":         shared.current_q.tolist(),
                        "dq":        shared.current_dq.tolist(),
                        "ee_pos":    shared.current_ee_pos.tolist(),
                        "ee_rotvec": shared.current_ee_rotvec.tolist(),
                        "gripper":   shared.gripper_width,
                        "timestamp": time.monotonic(),
                    }
                sock.send_json(reply)

            elif msg_type == "set_target":
                try:
                    new_q = np.array(msg["q"], dtype=float)
                    if len(new_q) != 7:
                        sock.send_json({"status": "error", "message": "q must have 7 elements"})
                        continue
                    with shared.lock:
                        cur_q = shared.current_q.copy()
                        delta = np.clip(new_q - cur_q, -config.max_delta_per_msg, config.max_delta_per_msg)
                        shared.target_q = cur_q + delta
                    sock.send_json({"status": "ok", "timestamp": time.monotonic()})
                except (KeyError, ValueError) as e:
                    sock.send_json({"status": "error", "message": str(e)})

            elif msg_type == "gripper_move":
                width = float(msg.get("width", 0.08))
                speed = float(msg.get("speed", 0.1))
                # TODO: replace with actual Gripper.move() call once API is confirmed
                with shared.lock:
                    shared.gripper_width = width
                logger.info(f"Gripper move: width={width:.4f}m speed={speed:.3f}m/s (stub)")
                sock.send_json({"status": "ok", "timestamp": time.monotonic()})

            else:
                sock.send_json({"status": "error", "message": f"Unknown type: {msg_type!r}"})

    finally:
        sock.close()
        ctx.term()
        logger.info("ZMQ server closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Franka ZMQ Server: joint-hold loop + ZMQ state relay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--stiffness", type=float, nargs=7, default=[200.0] * 7,
        metavar=("K1", "K2", "K3", "K4", "K5", "K6", "K7"),
    )
    parser.add_argument(
        "--damping", type=float, nargs=7, default=None,
        metavar=("D1", "D2", "D3", "D4", "D5", "D6", "D7"),
        help="Default: 2*sqrt(stiffness).",
    )
    parser.add_argument("--max_delta_tau", type=float, default=1.0)
    parser.add_argument("--max_delta_per_msg", type=float, default=0.1)
    parser.add_argument("--no_goto_home", action="store_true")
    args = parser.parse_args()

    stiffness = np.array(args.stiffness)
    damping = 2.0 * np.sqrt(stiffness) if args.damping is None else np.array(args.damping)

    config = ServerConfig(
        stiffness=stiffness,
        damping=damping,
        max_delta_tau=args.max_delta_tau,
        max_delta_per_msg=args.max_delta_per_msg,
        goto_home=not args.no_goto_home,
    )
    shared = SharedState()

    def _on_signal(*_):
        logger.info("Shutdown signal received.")
        with shared.lock:
            shared.shutdown = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    control_thread = threading.Thread(
        target=control_loop_thread,
        args=(args.robot_ip, shared, config),
        daemon=True,
        name="franka_control",
    )
    control_thread.start()

    t0 = time.time()
    while True:
        with shared.lock:
            if shared.control_running:
                break
            if shared.shutdown or shared.error:
                logger.error(f"Control loop failed: {shared.error}")
                return 1
        if time.time() - t0 > 30.0:
            logger.error("Timed out waiting for control loop")
            return 1
        time.sleep(0.1)

    logger.info("Control loop ready. Starting ZMQ server...")
    zmq_server_loop(shared, args.port, config)

    control_thread.join(timeout=5.0)
    if control_thread.is_alive():
        logger.warning("Control thread did not exit in time")
    logger.info("Server shut down cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
