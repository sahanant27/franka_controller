# pylibfranka — reference

**Source of truth:** https://github.com/frankarobotics/libfranka/tree/main/pylibfranka
(bindings in `src/*.cpp`, API docs in `docs/api/*.rst`, examples in `examples/`).
The controllers in this repo are adaptations of those examples. When in doubt,
check the official source — this file mirrors it for quick comparison.

## Two control modalities (map to our two needs)

| Need | Modality | How | RT loop runs in |
|---|---|---|---|
| **Movement** (point-to-point) | **Async position control** | `AsyncPositionControlHandler` — set joint targets at ~50 Hz | compiled C++ (robust) |
| **Joint impedance** (compliant/contact) | **Torque control** | `start_torque_control()` + `readOnce/writeOnce` with our impedance law | Python loop (use `kIgnore`) |

Switch between them per phase: async-position to move to a pre-grasp pose, then
torque-impedance for the compliant contact phase. Plus the **Gripper** API to grasp.

## API cheat-sheet (from the bindings)

```python
import pylibfranka as franka
robot = franka.Robot(ip, franka.RealtimeConfig.kIgnore)   # kIgnore tolerates Python loop jitter
robot.read_once()            -> RobotState
robot.load_model()           -> Model
robot.start_torque_control() -> ActiveControl            # for the torque/impedance loop
robot.set_collision_behavior(lo_tau, hi_tau, lo_F, hi_F)
robot.set_joint_impedance([k1..k7])       # internal impedance gains (used by built-in modes)
robot.set_cartesian_impedance([k1..k6])
robot.set_EE(...) / set_K(...) / set_load(...)
robot.automatic_error_recovery()
robot.stop()

# torque/impedance loop:
active = robot.start_torque_control()
state, duration = active.readOnce()       # duration.to_sec()
active.writeOnce(franka.Torques(tau_list)) # also: JointPositions/JointVelocities/CartesianPose/CartesianVelocities
# Torques(tau_J=[7]).motion_finished = bool

# Model: model.mass(state)->49, model.coriolis(state)->7, model.gravity(state)->7,
#        model.zero_jacobian(state)->42  (see docs/api/model.rst for pose/body jacobians)

# RobotState fields: q, dq, tau_J, tau_ext_hat_filtered (contact torque),
#   O_T_EE (16, column-major), O_F_ext_hat_K (external wrench), robot_mode, time, ...

# Gripper (separate connection):
g = franka.Gripper(ip)
g.homing(); g.read_once() -> GripperState(width, max_width, is_grasped, temperature)
g.move(width, speed); g.grasp(width, speed, force, ...); g.stop()
```

Enums: `RealtimeConfig.{kEnforce,kIgnore}`, `ControllerMode.{JointImpedance,CartesianImpedance}`,
`RobotMode.{Idle,Move,Guiding,Reflex,UserStopped,AutomaticErrorRecovery,Other}`.

---

## GOLDEN: async position control (movement) — official example

The reference movement pattern. Configure the handler, then push joint-position
targets at ~50 Hz; the handler runs the high-rate motion internally. Our
movement controller should match this shape.

```python
# https://github.com/frankarobotics/libfranka/blob/main/pylibfranka/examples/async_position_control.py
import signal, sys, time, math, argparse, threading
from datetime import timedelta
import pylibfranka as franka
from example_common import setDefaultBehaviour

kDefaultMaximumVelocities = [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
kDefaultGoalTolerance = 10.0
motion_finished = False

def signal_handler(sig, frame):
    global motion_finished
    if sig == signal.SIGINT:
        motion_finished = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)

    try:
        robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        sys.exit(-1)

    setDefaultBehaviour(robot)

    initial_position = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]
    time_elapsed = 0.0
    direction = 1.0
    time_since_last_log = 0.0

    def calculate_joint_position_target(period_sec):
        nonlocal time_elapsed, direction, time_since_last_log
        time_elapsed += period_sec
        target_positions = [initial_position[i] + direction * 0.25 for i in range(7)]
        time_since_last_log += period_sec
        if time_since_last_log >= 1.0:
            direction *= -1.0
            time_since_last_log = 0.0
        return franka.AsyncPositionControlHandler.JointPositionTarget(
            joint_positions=target_positions)

    joint_position_control_configuration = \
        franka.AsyncPositionControlHandler.Configuration(
            maximum_joint_velocities=kDefaultMaximumVelocities,
            goal_tolerance=kDefaultGoalTolerance)

    result = franka.AsyncPositionControlHandler.configure(
        robot, joint_position_control_configuration)
    if result.error_message is not None:
        print(result.error_message); sys.exit(-1)

    position_control_handler = result.handler
    target_feedback = position_control_handler.get_target_feedback()

    time_step = 0.020  # 20 ms, 50 Hz

    global motion_finished
    while not motion_finished:
        loop_start = time.monotonic()
        target_feedback = position_control_handler.get_target_feedback()
        if target_feedback.error_message is not None:
            print(target_feedback.error_message); sys.exit(-1)

        next_target = calculate_joint_position_target(time_step)
        command_result = position_control_handler.set_joint_position_target(next_target)
        if command_result.error_message is not None:
            print(command_result.error_message); sys.exit(-1)

        if time_elapsed > 10.0:
            position_control_handler.stop_control()
            motion_finished = True
            print("Control finished"); break

        sleep_time = time_step - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
```

Key handler API:
- `franka.AsyncPositionControlHandler.configure(robot, Configuration(maximum_joint_velocities=[7], goal_tolerance))` → result with `.handler`, `.error_message`
- `handler.set_joint_position_target(JointPositionTarget(joint_positions=[7]))` → `.motion_uuid`, `.was_successful`, `.error_message`
- `handler.get_target_feedback()` → `.status`, `.error_message`
- `handler.stop_control()`

## GOLDEN: torque / joint-impedance loop

The reference impedance pattern is `examples/joint_impedance_example.py` (official)
and our `joint_impedance.py` / `goto_home.py`: `start_torque_control()` →
`readOnce()` → `tau = -K·(q-q_target) - D·dq + coriolis` → `writeOnce(Torques(tau))`,
with `q_target` from a minimum-jerk trajectory. Use this when contact compliance
matters; use async position control for plain moves.
