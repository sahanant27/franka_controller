#!/usr/bin/env python3
"""Fresh RT-server — async joint-position control over zmq (ROBOT PC).

Owns the robot, a JointPositionController (async handler), and a TargetStreamer
(50 Hz feeder thread that keeps async control alive — see joint_position_controller.py
for why a single set_target does NOT hold). The zmq REP loop runs in the main
thread and only touches the streamer's thread-safe methods.

Protocol (JSON over zmq REQ/REP):
  {"cmd":"ping"}                -> {"ok":true}
  {"cmd":"get_state"}           -> {"ok":true,"q":[7],"dq":[7],"ee_pose":[[4]x4]}
  {"cmd":"set_target","q":[7]}  -> {"ok":true}     (updates the streamed target)
  {"cmd":"recover"}             -> {"ok":true}     (reflex/collision recovery: clear the
                                   error, re-arm the controller, hold current pose)
  {"cmd":"gripper","action":"open"|"close"[,"width","force"]}
                                -> {"ok":true,"width":<mean finger pos>}  (returns
                                   IMMEDIATELY; the open/close runs in the background so
                                   the 50 Hz state polling never stalls)

get_state additionally reports "gripper": mean finger position [m] (width/2,
~0.04 = fully open), or null with --no-gripper.

Run on the ROBOT PC (e-stop in hand — the arm becomes live on startup):
  python servers/franka_server.py --ip 172.16.0.2 --bind tcp://0.0.0.0:5556
"""
import argparse
import os
import sys
import threading
import time
import traceback

import zmq
import pylibfranka as franka

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `controllers`
from controllers.home import set_gripper
from controllers.joint_position_controller import JointPositionController, TargetStreamer


class GripperService:
    """Gripper access that never stalls the REP loop.

    Reads: a background thread with its OWN gripper connection caches the width
    (~10 Hz) so get_state can attach it for free. Commands: executed on a
    worker thread via home.set_gripper (which opens its own connection per
    call, the pattern already validated there) — the zmq reply returns
    immediately, mirroring the ROS stack's send_goal_async semantics.
    """

    def __init__(self, ip, poll_s=0.1):
        self.ip = ip
        self._lock = threading.Lock()
        self._width = None
        self._busy = False
        self._running = True
        threading.Thread(target=self._poll, args=(poll_s,), daemon=True,
                         name="gripper_poll").start()

    def _poll(self, poll_s):
        g = franka.Gripper(self.ip)
        while self._running:
            try:
                w = float(g.read_once().width)
                with self._lock:
                    self._width = w
            except Exception:
                pass                              # transient read hiccup: keep last width
            time.sleep(poll_s)

    def mean_finger(self):
        """Mean finger position [m] = width/2 (matches /joint_states[7]); None until read."""
        with self._lock:
            return None if self._width is None else self._width / 2.0

    def command(self, action, width=0.0, force=40.0):
        """Fire-and-forget open/close on a worker thread. One at a time."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def run():
            try:
                set_gripper(self.ip, action=action, width=width, force=force)
            except Exception as e:
                print(f"gripper {action} failed: {e}")
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=run, daemon=True, name="gripper_cmd").start()
        return True

    def stop(self):
        self._running = False


def handle(req, streamer, grip):
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "get_state":
        s = streamer.get_state()
        rep = {"ok": True, "q": s["q"].tolist(), "dq": s["dq"].tolist(),
               "ee_pose": s["T_base_ee"].tolist(),
               "gripper": grip.mean_finger() if grip else None,
               "controlling": s["controlling"]}   # False in programming/guiding mode
        if "error" in s:                          # the state-READ loop died (state is stale)
            rep["ok"] = False
            rep["error"] = f"streamer error: {s['error']}"
        return rep
    if cmd == "set_target":
        streamer.update_target(req["q"])      # clamps + bounds happen in the feeder
        return {"ok": True}
    if cmd == "gripper":
        if grip is None:
            return {"ok": False, "error": "gripper disabled (--no-gripper)"}
        started = grip.command(req.get("action", "open"),
                               width=float(req.get("width", 0.0)),
                               force=float(req.get("force", 40.0)))
        return {"ok": True, "started": started, "width": grip.mean_finger()}
    return {"ok": False, "error": f"unknown cmd: {cmd!r}"}


def arm_streamer(robot, args):
    """Configure the async controller + start the 50 Hz feeder (holds current pose)."""
    ctrl = JointPositionController(robot, max_velocity=args.max_vel, goal_tolerance=args.tol)
    streamer = TargetStreamer(ctrl, rate_hz=args.rate)
    streamer.start()
    return streamer


def recover(robot, streamer, args):
    """Reflex/collision/e-stop recovery: release the (dead) controller, clear the
    robot error, re-arm fresh. The new streamer holds wherever the arm now is."""
    try:
        streamer.stop()                       # joins feeder; releases control if any
    except Exception:
        pass                                  # control may already be gone (reflex)
    robot.automatic_error_recovery()
    return arm_streamer(robot, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="172.16.0.2")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max-vel", type=float, default=0.4, help="max joint velocity [rad/s]")
    ap.add_argument("--tol", type=float, default=0.05, help="goal tolerance [rad]")
    ap.add_argument("--rate", type=float, default=50.0, help="feeder re-send rate [Hz]")
    ap.add_argument("--no-gripper", action="store_true",
                    help="no Franka Hand attached / skip gripper support")
    args = ap.parse_args()

    robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    streamer = arm_streamer(robot, args)      # arm now actively held at current pose
    grip = None if args.no_gripper else GripperService(args.ip)
    print(f"streamer running @ {args.rate} Hz (holding current pose)"
          + ("" if grip else " — gripper disabled"))

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 500)        # so Ctrl-C is responsive
    sock.bind(args.bind)
    print(f"franka_server listening on {args.bind}  (Ctrl-C to stop)")

    warned_error = False
    try:
        while True:
            try:
                req = sock.recv_json()
            except zmq.Again:
                if streamer._error and not warned_error:   # dead read loop: stay up —
                    print(f"streamer error (recoverable via 'recover'): {streamer._error}")
                    warned_error = True                    # the client can send recover
                continue
            try:
                if req.get("cmd") == "recover":
                    streamer = recover(robot, streamer, args)
                    warned_error = False
                    print("recovered: error cleared, controller re-armed")
                    rep = {"ok": True}
                else:
                    rep = handle(req, streamer, grip)
            except Exception as e:            # per-request boundary: always reply
                rep = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            sock.send_json(rep)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        streamer.stop()                       # stop feeder + release async control
        if grip is not None:
            grip.stop()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
