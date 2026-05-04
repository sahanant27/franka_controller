#!/usr/bin/env python3

"""
Franka ZMQ Bridge — runs on Computer A (robot computer)

Sits between the policy client on Computer B and the joint-hold controller
(franka_zmq_server.py) running locally.

Architecture:
    Computer B (franka_zmq_robot.py)
        ↕  ZMQ REQ/REP  (--listen_port, default 5556, network-facing)
    run_robot.py  ← this file
        ↕  ZMQ REQ/REP  (--controller_port, default 5555, localhost only)
    franka_zmq_server.py

Responsibilities:
  - Exposes the same message protocol as franka_zmq_server.py to the network
  - Shields the real-time control process from direct network access
  - Adds delta-EE integration (--delta_ee) on top of joint-hold controller
  - Connection health monitoring and reconnect on timeout

Message protocol (identical to franka_zmq_server.py — transparent pass-through):
    get_state     → {"q":[7], "dq":[7], "ee_pos":[3], "ee_rotvec":[3],
                     "gripper":float, "timestamp":float}
    set_target    {"q":[7]}                            → {"status":"ok", ...}
    set_ee_target {"pos":[3], "rotvec":[3]}            → {"status":"ok", ...}
                  With --delta_ee: values treated as deltas and integrated here.
                  NOTE: without --delta_ee, requires an OSC-capable controller.
    gripper_move  {"width":float, "speed":float}       → {"status":"ok", ...}

Usage:
    # Start franka_zmq_server.py first (binds to localhost:5555):
    python franka_zmq_server.py --robot_ip 172.16.0.2 --port 5555

    # Then start this bridge (listens on 0.0.0.0:5556 for Computer B):
    python run_robot.py --listen_port 5556

    # On Computer B, point franka_zmq_robot.py at this machine's IP on port 5556.
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

from utils.transforms import axis_angle_to_rot_matrix, rot_matrix_to_axis_angle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_robot")


# ---------------------------------------------------------------------------
# Controller client — connects to franka_zmq_server.py (local)
# ---------------------------------------------------------------------------

class ControllerClient:
    """ZMQ REQ client for franka_zmq_server.py running on localhost."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 2000):
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context()
        self._sock: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.connect(f"tcp://{self._host}:{self._port}")
        logger.debug(f"Controller client connected to tcp://{self._host}:{self._port}")

    def reconnect(self) -> None:
        logger.warning("Reconnecting to controller...")
        self._connect()

    def request(self, msg: dict, max_retries: int = 3) -> dict:
        for attempt in range(max_retries):
            try:
                self._sock.send_json(msg)
                return self._sock.recv_json()
            except zmq.Again:
                logger.warning(
                    f"Controller timeout on {msg.get('type')!r} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    self.reconnect()
        raise ConnectionError(
            f"franka_zmq_server not responding after {max_retries} attempts "
            f"at tcp://{self._host}:{self._port}"
        )

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._ctx.term()


# ---------------------------------------------------------------------------
# Delta-EE state tracker (bridge-side only — no libfranka model needed)
# ---------------------------------------------------------------------------

@dataclass
class BridgeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Last known EE state from controller (refreshed on every get_state)
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shutdown: bool = False


# ---------------------------------------------------------------------------
# Bridge loop
# ---------------------------------------------------------------------------

def bridge_loop(
    controller: ControllerClient,
    state: BridgeState,
    listen_port: int,
    delta_ee: bool,
) -> None:
    """Main thread: REP socket for Computer B, proxies to controller."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s poll for shutdown check
    sock.bind(f"tcp://*:{listen_port}")
    logger.info(f"Bridge REP socket listening on tcp://*:{listen_port}")

    try:
        while True:
            with state.lock:
                if state.shutdown:
                    break

            # --- receive from Computer B ---
            try:
                msg_bytes = sock.recv()
            except zmq.Again:
                continue  # timeout — recheck shutdown flag

            try:
                msg = json.loads(msg_bytes)
            except json.JSONDecodeError as e:
                sock.send_json({"status": "error", "message": f"JSON decode: {e}"})
                continue

            msg_type = msg.get("type")

            # --- proxy or transform, then send reply ---
            try:
                reply = _handle(msg_type, msg, controller, state, delta_ee)
            except ConnectionError as e:
                reply = {"status": "error", "message": str(e)}
            except Exception as e:
                reply = {"status": "error", "message": f"Bridge error: {e}"}

            sock.send_json(reply)

    finally:
        sock.close()
        ctx.term()
        logger.info("Bridge socket closed.")


def _handle(
    msg_type: str,
    msg: dict,
    controller: ControllerClient,
    state: BridgeState,
    delta_ee: bool,
) -> dict:
    """Route one message: transparent proxy or delta-EE integration."""

    if msg_type == "get_state":
        reply = controller.request({"type": "get_state"})
        # Keep a local copy of EE state for delta integration
        with state.lock:
            state.ee_pos = np.array(reply["ee_pos"])
            state.ee_rotvec = np.array(reply["ee_rotvec"])
        return reply

    elif msg_type == "set_target":
        # Transparent joint-target proxy
        return controller.request({"type": "set_target", "q": msg["q"]})

    elif msg_type == "set_ee_target":
        pos = np.array(msg["pos"], dtype=float)
        rotvec = np.array(msg["rotvec"], dtype=float)

        if delta_ee:
            # Integrate delta on bridge using last known EE state
            with state.lock:
                cur_pos = state.ee_pos.copy()
                cur_rotvec = state.ee_rotvec.copy()
            abs_pos = cur_pos + pos
            abs_rot = axis_angle_to_rot_matrix(rotvec) @ axis_angle_to_rot_matrix(cur_rotvec)
            abs_rotvec = rot_matrix_to_axis_angle(abs_rot)
        else:
            abs_pos = pos
            abs_rotvec = rotvec

        # Forward as absolute EE target to the controller.
        # franka_zmq_server.py (joint-hold only) does not handle set_ee_target.
        # To enable EE control: add OSC torque handling to franka_zmq_server.py,
        # or implement IK here and convert to set_target {"q": [...7]}.
        reply = controller.request({
            "type": "set_ee_target",
            "pos": abs_pos.tolist(),
            "rotvec": abs_rotvec.tolist(),
        })
        if reply.get("status") == "error" and "Unknown type" in reply.get("message", ""):
            return {
                "status": "error",
                "message": (
                    "set_ee_target is not supported by the current controller. "
                    "Add OSC control to franka_zmq_server.py or use joint-space actions."
                ),
            }
        return reply

    elif msg_type == "gripper_move":
        return controller.request({
            "type": "gripper_move",
            "width": float(msg.get("width", 0.08)),
            "speed": float(msg.get("speed", 0.1)),
        })

    else:
        return {"status": "error", "message": f"Unknown type: {msg_type!r}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Franka ZMQ Bridge: proxies Computer B → franka_zmq_server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--controller_ip", default="127.0.0.1",
        help="IP of franka_zmq_server (usually localhost on Computer A).",
    )
    parser.add_argument(
        "--controller_port", type=int, default=5555,
        help="Port of franka_zmq_server.",
    )
    parser.add_argument(
        "--listen_port", type=int, default=5556,
        help="Port this bridge binds on for Computer B connections.",
    )
    parser.add_argument(
        "--controller_timeout_ms", type=int, default=2000,
        help="ZMQ receive timeout (ms) toward franka_zmq_server.",
    )
    parser.add_argument(
        "--delta_ee", action="store_true",
        help=(
            "Treat set_ee_target values as deltas. "
            "Bridge integrates them using the last EE state from get_state."
        ),
    )
    args = parser.parse_args()

    controller = ControllerClient(
        host=args.controller_ip,
        port=args.controller_port,
        timeout_ms=args.controller_timeout_ms,
    )

    # Verify controller is reachable before opening the network-facing socket
    try:
        state_resp = controller.request({"type": "get_state"})
        logger.info(
            f"Controller reachable — "
            f"q={[f'{v:.3f}' for v in state_resp['q']]}"
        )
    except ConnectionError as e:
        logger.error(f"Cannot reach franka_zmq_server: {e}")
        controller.close()
        return 1

    bridge_state = BridgeState(
        ee_pos=np.array(state_resp["ee_pos"]),
        ee_rotvec=np.array(state_resp["ee_rotvec"]),
    )

    def _on_signal(*_):
        logger.info("Shutdown signal received.")
        with bridge_state.lock:
            bridge_state.shutdown = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if args.delta_ee:
        logger.info("Delta-EE mode enabled: set_ee_target values integrated as deltas.")

    try:
        bridge_loop(controller, bridge_state, args.listen_port, args.delta_ee)
    finally:
        controller.close()

    logger.info("Bridge shut down cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
