import argparse
from dataclasses import dataclass, field

import numpy as np

from pylibfranka import CartesianPose

from controller.base_controller import BaseController, ControllerConfig
from utils.control import log_pose_error, pose_error_norms
from utils.motion import CartesianTargetPlanner
from utils.transforms import compute_pose_error, create_frame_from_xyzrpy


@dataclass
class CartesianPoseCfg:
    controller: ControllerConfig = field(
        default_factory=lambda: ControllerConfig(control_mode="cpose", auto_goto_home=True)
    )
    trajectory_duration: float = 8.0
    translation_error_threshold: float = 5e-3
    rotation_error_threshold: float = 1e-2
    planner_duration: float = 3.0
    log_interval: float = 1.0


class CartesianPoseController(BaseController):
    def __init__(self, robot_ip: str, config: CartesianPoseCfg | None = None):
        self.cfg = config if config is not None else CartesianPoseCfg()
        super().__init__(robot_ip, self.cfg.controller)

        self._target_planner = CartesianTargetPlanner(default_duration=self.cfg.planner_duration)
        self._planned_target = None
        self._goal_pose = None
        self.time_elapsed = 0.0
        self.motion_finished = False
        self._timed_out = False
        self._next_log_time = 0.0

    def write_command(self, pose_4x4: np.ndarray, motion_finished: bool = False):
        pose_flat = np.array(pose_4x4).flatten(order="F").tolist()
        cmd = CartesianPose(pose_flat)
        cmd.motion_finished = motion_finished
        self.active_control.writeOnce(cmd)
        return cmd

    def task_error(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        return compute_pose_error(current_pose, target_pose)

    def _goal_reached(self, error_6d: np.ndarray) -> bool:
        translation_error, rotation_error, _ = pose_error_norms(error_6d)
        return (
            translation_error < self.cfg.translation_error_threshold
            and rotation_error < self.cfg.rotation_error_threshold
        )

    def set_target(self, abs_target=None, delta_target=None, duration=None) -> np.ndarray:
        self._update_state()
        goal_pose = self._target_planner.set_goal(
            current_pose=self._cartesian_pose,
            abs_target=abs_target,
            delta_target=delta_target,
            duration=duration,
        )
        self._goal_pose = goal_pose.copy()
        self._planned_target = self._target_planner.current_target.copy()
        self.motion_finished = False
        self.time_elapsed = 0.0
        self._timed_out = False
        self._next_log_time = 0.0
        return goal_pose

    def _control_step(self) -> None:
        if self._target_planner.current_target is None:
            raise ValueError("Planner has no target. Call set_target(...) before run().")

        self._update_state()
        self.time_elapsed += self.duration.to_sec()
        self._planned_target = self._target_planner.step(self.duration.to_sec())

        planned_error_6d = self.task_error(self._cartesian_pose, self._planned_target)
        final_error_6d = self.task_error(self._cartesian_pose, self._target_planner.goal_pose)
        planned_norms = pose_error_norms(planned_error_6d)
        final_norms = pose_error_norms(final_error_6d)

        planner_finished = self._target_planner.is_finished()
        reached_goal = planner_finished and self._goal_reached(final_error_6d)
        self._timed_out = self.time_elapsed >= self.cfg.trajectory_duration
        self.motion_finished = reached_goal or self._timed_out

        if self.time_elapsed >= self._next_log_time:
            log_pose_error("planned_error", planned_norms, time_elapsed=self.time_elapsed, include_total=False)
            log_pose_error("goal_error", final_norms, time_elapsed=self.time_elapsed, include_total=False)
            self._next_log_time += self.cfg.log_interval

        self.write_command(self._planned_target, motion_finished=self.motion_finished)

    def run(self, abs_target=None, delta_target=None, duration=None) -> np.ndarray:
        goal_pose = self.set_target(abs_target=abs_target, delta_target=delta_target, duration=duration)
        print(f"Starting Cartesian pose motion with timeout {self.cfg.trajectory_duration:.2f}s")
        while not self.motion_finished:
            self._control_step()
        final_error_6d = self.task_error(self._cartesian_pose, self._goal_pose)
        final_norms = pose_error_norms(final_error_6d)
        if self._timed_out:
            print(f"Motion timed out at t={self.time_elapsed:.2f}s")
        else:
            print(f"Motion finished at t={self.time_elapsed:.2f}s")
        log_pose_error("final_goal_error", final_norms, time_elapsed=self.time_elapsed, include_total=False)
        return goal_pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--droll", type=float, default=0.0)
    parser.add_argument("--dpitch", type=float, default=0.0)
    parser.add_argument("--dyaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    controller = None
    try:
        controller = CartesianPoseController(args.ip)
        delta_target = create_frame_from_xyzrpy(
            xyz=(args.dx, args.dy, args.dz),
            rpy=(np.deg2rad(args.droll), np.deg2rad(args.dpitch), np.deg2rad(args.dyaw)),
        )
        controller.run(delta_target=delta_target, duration=args.duration)
    except Exception as exc:
        print(f"Error: {exc}")
        if controller is not None:
            controller.robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    main()
