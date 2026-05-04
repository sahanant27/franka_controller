import argparse
from dataclasses import dataclass, field

import numpy as np

from pylibfranka import JointPositions

from controller.base_controller import BaseController, ControllerConfig
from utils.motion import JointTargetPlanner


@dataclass
class JointPositionCfg:
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(control_mode="jpose", auto_goto_home=True)
    )
    trajectory_duration: float = 8.0
    joint_error_threshold: float = 1e-3   # rad — per-joint convergence tolerance
    planner_duration: float = 3.0
    log_interval: float = 1.0


class JointPositionController(BaseController):
    def __init__(self, robot_ip: str, config: JointPositionCfg | None = None):
        self.cfg = config if config is not None else JointPositionCfg()
        super().__init__(robot_ip, self.cfg.controller)

        self._target_planner = JointTargetPlanner(default_duration=self.cfg.planner_duration)
        self._planned_target = None
        self._goal_q = None
        self.time_elapsed = 0.0
        self.motion_finished = False
        self._timed_out = False
        self._next_log_time = 0.0

    def write_command(self, q, motion_finished: bool = False):
        cmd = JointPositions(np.asarray(q).tolist())
        cmd.motion_finished = motion_finished
        self.active_control.writeOnce(cmd)
        return cmd

    def _goal_reached(self, q_error: np.ndarray) -> bool:
        return np.all(np.abs(q_error) < self.cfg.joint_error_threshold)

    def set_target(self, q_target, duration=None) -> np.ndarray:
        self._update_state()
        goal_q = self._target_planner.set_goal(
            current_q=self._joint_pose,
            target_q=np.array(q_target, dtype=float),
            duration=duration,
        )
        self._goal_q = goal_q.copy()
        self._planned_target = self._target_planner.current_target.copy()
        self.motion_finished = False
        self.time_elapsed = 0.0
        self._timed_out = False
        self._next_log_time = 0.0
        return goal_q

    def _control_step(self) -> None:
        if self._target_planner.current_target is None:
            raise ValueError("Planner has no target. Call set_target(...) before run().")

        self._update_state()
        self.time_elapsed += self.duration.to_sec()
        self._planned_target = self._target_planner.step(self.duration.to_sec())

        q_error = self._goal_q - self._joint_pose

        planner_finished = self._target_planner.is_finished()
        reached_goal = planner_finished and self._goal_reached(q_error)
        self._timed_out = self.time_elapsed >= self.cfg.trajectory_duration
        self.motion_finished = reached_goal or self._timed_out

        if self.time_elapsed >= self._next_log_time:
            print(f"t={self.time_elapsed:.2f}s | joint_error={np.linalg.norm(q_error):.6f} rad")
            self._next_log_time += self.cfg.log_interval

        self.write_command(self._planned_target, motion_finished=self.motion_finished)

    def run(self, q_target, duration=None) -> np.ndarray:
        goal_q = self.set_target(q_target, duration=duration)
        print(f"Starting joint position motion with timeout {self.cfg.trajectory_duration:.2f}s")
        while not self.motion_finished:
            self._control_step()
        q_error = self._goal_q - self._joint_pose
        if self._timed_out:
            print(f"Motion timed out at t={self.time_elapsed:.2f}s | final_error={np.linalg.norm(q_error):.6f} rad")
        else:
            print(f"Motion finished at t={self.time_elapsed:.2f}s | final_error={np.linalg.norm(q_error):.6f} rad")
        return goal_q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2")
    parser.add_argument(
        "--q", type=float, nargs=7,
        default=[0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Target joint angles (rad).",
    )
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    controller = None
    try:
        controller = JointPositionController(args.ip)
        controller.run(q_target=args.q, duration=args.duration)
    except Exception as exc:
        print(f"Error: {exc}")
        if controller is not None:
            controller.robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    main()
