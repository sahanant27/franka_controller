# Operation Space Controller Debug Findings

## Goal
Understand why the current Franka operation-space controller can move, but does not converge cleanly to Cartesian targets, especially for orientation.

## Current Setup
- Controller file: [operation_space.py](/home/anant/projects/franka_controller/operation_space.py)
- Base controller: [base_controller.py](/home/anant/projects/franka_controller/base_controller.py)
- Pose error helper: [utils/transforms.py](/home/anant/projects/franka_controller/utils/transforms.py)
- Target planner: [utils/motion.py](/home/anant/projects/franka_controller/utils/motion.py)

## Confirmed Facts
- Pose source is `O_T_EE`, so pose is in the base/world frame.
- Jacobian source is `model.zero_jacobian(robot_state)`, which maps to the end-effector zero Jacobian in the base/world frame.
- Torque rate limiting and hard torque clipping were both inactive during the tested `+0.10 m` x-translation runs.
- Planner behavior is not the main issue. After planner completion, the controller still sits at a steady nonzero residual error.

## Changes Tested

### 1. Added Coriolis compensation
- Result: no meaningful improvement in convergence.

### 2. Increased rotational gains
- Result: no meaningful improvement in final residual error.

### 3. Added small damping to task-space inverse
- Change: `lambda_inv_damped = lambda_inv + 1e-3 * I`
- Result: no meaningful improvement.

### 4. Changed torque-rate limit reference
- Switched to `robot_state.tau_J_d` instead of previous commanded torque.
- Result: no meaningful improvement.

### 5. Updated orientation error convention
- Switched `compute_pose_error(...)` to a Franka-style quaternion/sign/base-frame formulation.
- Result: real improvement.
- Translation-only drift improved roughly from:
  - rotation residual `~0.123 rad`
  - to `~0.066 rad`

## Key Logging Findings

### For `full` task mode on `+0.10 m` x target
- Final steady residual after planner completion:
  - translation error `~0.0063 to 0.0066 m`
  - rotation error `~0.066 rad`
- `des_acc` remains nonzero even after motion stalls.
- `eef_velocity` becomes very small.
- Interpretation:
  - controller is still commanding correction,
  - but robot is not reducing the residual error further.

### Torque / safety logs
- `rate_limit_delta_norm = 0`
- `clip_delta_norm = 0`
- Interpretation:
  - safety layers are not what is preventing convergence in the tested case.

### Task-space mapping logs
- `des_acc_angular` is dominated by the z component.
- `task_wrench_angular` is not dominated by z; it is heavily mixed into other axes.
- Interpretation:
  - task-space inertia mapping is coupling rotational axes strongly.

## Decoupled Task Tests

### `translation_only`
- Rotational task error, rotational damping term, and rotational desired acceleration were zeroed.
- Despite that:
  - `task_wrench_angular_norm` remained nonzero
  - physical orientation drift still occurred (`~0.063 rad`)
- Interpretation:
  - translational commands are being converted into angular wrench by the full `Lambda` mapping.

### `rotation_only`
- Translational task error, translational damping term, and translational desired acceleration were zeroed.
- Despite that:
  - `task_wrench_linear_norm` remained large
  - robot moved violently in translation/downward direction
- Test was stopped manually with emergency stop.
- Physical observation:
  - the main visible motion was dominated by **joint 2 moving downward**
  - the rest of the arm changed much less by comparison
- Interpretation:
  - rotational commands are being converted into linear wrench by the full `Lambda` mapping.
  - the unintended motion may be projecting strongly into a specific joint-space direction, not only as a diffuse whole-arm effect.

## Strongest Current Conclusion
The main remaining issue is very likely **translation/rotation coupling inside the operational-space inertia mapping**:

`task_wrench = inv(J M^-1 J^T) @ des_acc`

This coupling is strong enough that:
- pure translation command does not remain pure in wrench space
- pure rotation command does not remain pure in wrench space

This is the clearest explanation for the observed drift and dangerous rotation-only behavior.

## Things Likely Not To Blame
- Planner interpolation
- Torque rate limiting
- Hard torque clipping
- Small task-space damping change alone
- Gain magnitude alone

## Things That Did Matter
- Orientation error convention

## Best Next Steps
1. Inspect the 6x6 `Lambda` matrix directly, especially off-diagonal translation/rotation blocks.
2. Try a safer decoupled approximation:
   - block-diagonal translational/rotational `Lambda`, or
   - separate translational and rotational solves.
3. Compare against a simpler Jacobian-transpose Cartesian impedance baseline:
   - `tau = J.T @ (-K e - D J dq) + coriolis`
4. Avoid rerunning `rotation_only` without extra safety or smaller targets.
5. If more logging is added later, log `tau_raw` per joint to check whether joint 2 consistently receives the dominant unintended command.

## Practical Restart Note
If starting fresh next session, begin from:
- current orientation-error convention
- current logging setup
- focus immediately on `Lambda` coupling and block structure
