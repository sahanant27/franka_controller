# Streamed 1 kHz position control — why, what exists, and how to bring it up

The goal: reproduce the ROS stack's smooth trajectory-following on the pylibfranka
stack. Everything is already written; ONE open issue remains (the 1 ms deadline,
§4). This doc is the map for fixing and verifying it.

---

## 1. The problem with the async controller (what we run today)

`AsyncPositionControlHandler` is a point-to-point GOAL-SEEKER. Its own source
defines success as *arrive and stop* (libfranka
`src/async_control/async_position_control_handler.cpp`):

```cpp
if (position_error < goal_tolerance_ && current_speed < kTargetReachedVelocityThreshold)
```

Fed a continuous stream of policy targets it perpetually plans-brakes-replans:
- sparse targets (10 Hz)  → visible reach-brake-restart ("start-stop")
- dense micro-targets     → tremble/judder around the commanded path
- confirmed by a policy-free probe: even a pure slow sine rang
  (`lfo_inference/scripts/test/stream_probe.py`)

No client-side smoothing can fix this: the braking is planned in firmware after
our last touchpoint. The async controller stays for what it is good at —
homing, calibration, discrete moves.

## 2. What ROS actually does (with receipts)

The lerobot-era stack (franka_ros2) that felt smooth:

```
policy (10 Hz, one target per msg)
  └► JointTrajectoryController          [fr3_ros_controllers.yaml:
        splines from CURRENT reference   type: joint_trajectory_controller/...,
        (pos+vel) to each new target;    1000 Hz update rate]
        samples the spline at 1 kHz
      └► command interface
            effort variant: JTC's PID (p 50-600, d 5-30) → torques
            position variant: firmware impedance tracks the reference
          └► franka_hardware/src/robot.cpp:
                startJointPositionControl(ControllerMode::kJointImpedance)
                active_control_->writeOnce(...)   // every 1 ms
```

Key insight: the policy's 10 Hz staircase NEVER reaches the robot — only the
1 kHz spline through it does, velocity-continuous at every target switch. The
"PD" of the position variant lives in the FIRMWARE (`kJointImpedance` mode).

## 3. The pylibfranka substitute (already built)

pylibfranka binds the exact same libfranka entry points franka_hardware uses
(`Robot::startJointPositionControl` → `ActiveMotionGenerator<JointPositions>`
→ `writeOnce`, all in libfranka `src/robot.cpp` / `src/active_control.cpp`):

| ROS piece | our substitute |
|---|---|
| JTC spline sampling | critically-damped tracker (closed-form, C1, no overshoot) |
| 1 kHz read→update→write cycle | `readOnce → tracker step → writeOnce(JointPositions)` |
| hardware interface | `robot.start_joint_position_control(ControllerMode.JointImpedance)` |
| firmware impedance tracking | identical (same mode flag) |
| trajectory topic | existing zmq `set_target` (protocol unchanged) |
| RT-scheduled C++ process | **Python thread — the open issue (§4)** |

Code: `controllers/streamed_joint_position_controller.py`, selected with
`servers/franka_server.py --controller streamed` (default is `async`).
Knobs: `--tau` (reference time constant, default 0.06 s; bigger = smoother =
laggier) and `--max-vel` (reference velocity clamp).
Perception-PC side needs NO changes — same protocol, same port. Bonus: state
cache refreshes at 100 Hz-1 kHz instead of 30-50.

Not bound in pylibfranka (franka_ros2 optionally uses them): `franka::lowpassFilter`,
`franka::limitRate`. Both are few-line formulas; add to the 1 kHz loop if the
exact ROS safety envelope is wanted. The tracker already covers smoothing.

## 4. THE open issue: the 1 ms deadline

The robot demands a `writeOnce` response every 1 ms; too many misses trigger
the reflex `communication_constraints_violation` — which is exactly how the
first bring-up died. Their 1 kHz caller is RT-scheduled C++; ours is a Python
thread. Mitigations already in `franka_server.py` (streamed mode only):

- `sys.setswitchinterval(0.0005)` — default GIL slice is 5 ms(!); any other
  thread (zmq REP, gripper poll) could otherwise hold the GIL for five cycles
- `gc.freeze(); gc.disable()` — a gen-2 GC pause is multiple milliseconds
- `os.nice(-10)` (best effort)
- hot loop slimmed: state snapshot decimated to 100 Hz, minimal lock windows

### Bring-up procedure (in order)

```bash
# A. Measure the machine (NO robot needed, 10 s each):
python tools/rt_jitter.py
chrt -f 80 python tools/rt_jitter.py        # needs realtime group membership
#   prints kernel type (PREEMPT_RT?), overrun p99/p99.9/max, and a verdict.
#   Viable ~= zero overruns >2 ms and <0.1% >1 ms.

# B. If viable (likely only under chrt): start the server the same way:
chrt -f 80 python servers/franka_server.py --ip 172.16.0.2 --controller streamed
#   should hold pose silently — no reflex within the first minute is the first win.

# C. Acceptance test from the perception PC (policy-free sine):
python scripts/test/stream_probe.py --send
#   PASS: residual < ~20 mrad pk-pk, motion looks like a metronome, no stutter.
#   (async controller measurably rings on this same test — that is the baseline.)

# D. Only then: the real task, unchanged client:
python scripts/run_inference.py --proprio robot --task close_drawer --demo-num 1 --send
```

### Debug levers if reflexes persist

1. `chrt -f 90` + verify `ulimit -r` / membership in the `realtime` group
   (a libfranka install requirement anyway).
2. Kernel: `uname -r` — no PREEMPT_RT ⇒ worst-case latencies are unbounded;
   Franka's own setup guide requires an RT kernel for reliable 1 kHz.
3. Isolate load: stop the gripper poll (`--no-gripper`) and camera/browser/etc.
   processes on the robot PC while testing.
4. Raise `--tau` (gentler accelerations do not help deadlines, but rule out
   any motion-related aborts vs pure timing ones — read the reflex text).

## 4b. RealtimeConfig facts (verified from source — read before choosing a path)

`franka::RealtimeConfig` is **client-side only** (`libfranka/src/robot_impl.cpp`):
`kEnforce` = elevate the control thread to SCHED_FIFO **and require an RT
kernel**, throwing `RealtimeException` otherwise; `kIgnore` = skip that, warn,
continue. It does NOT relax the robot-side communication constraints — the
1 ms deadline and `communication_constraints_violation` reflex apply to BOTH
interfaces, always.

What differs between the interfaces is the COST of a late packet:

| | torque stream (impedance) | position stream (motion generator) |
|---|---|---|
| late packet effect | last torque briefly held — physically benign | reference discontinuity |
| extra validation | torque rate limit only (our slew limiter covers it) | kinematic continuity reflexes (velocity/accel/jerk discontinuity) on top of the comm watchdog |
| practical jitter tolerance | graceful — validated 1 kHz Python loop with kIgnore on THIS machine | brittle — reflexed within seconds on the same machine |

franka_ros2 auto-selects (`franka_hardware/src/robot.cpp`): `kEnforce` when
`hasRealtimeKernel()`, else `kIgnore`. On a standard Franka setup (PREEMPT_RT
mandated by the FCI docs) ROS therefore runs ENFORCE with an RT-scheduled
control thread — part of why its position-interface path is stable.

**Conclusion: on a non-RT machine, the torque (impedance) interface is the
jitter-tolerant one; the position interface effectively does require RT
scheduling.** The flag is not the mechanism — packet-loss cost is.

## 5. Chosen path: impedance executor, position-commanded

Decision: policy streaming moves to the **impedance server**
(`servers/impedance_server.py`, already validated on this hardware). To be
precise about what changes and what does not:

- **The command interface stays POSITION.** The policy still emits absolute
  joint targets; the client still streams position goals. Only the tracking
  mechanism changes: instead of the firmware motion generator (goal-seeker),
  a 1 kHz torque-PD tracks a server-side-ramped reference:
  `tau = -Kp (q - q_ref) - Kd dq (+ coriolis)`, gains fixed.
- This is the moral equivalent of the ROS effort-interface JTC (its PID:
  p 50-600, d 5-30 — same role, same ballpark as the impedance Kp/Kd).
- Client mapping: our smoothed/clamped `q_cmd` -> `set_action([dq, Kp, Kd])`
  with `dq = q_cmd - q_ref_current` (reference_mode per server config);
  gains constant from config. Server's `interp_time` ramp replaces the
  client interpolator's role.
- Consequences to expect: compliant tracking (contact-rich steps like the
  drawer push become gentler); steady-state tracking error under load scales
  with 1/Kp — tune gains if precision lags.
- Keep the async position controller for homing/resets (it is good at
  point-to-point), and `--controller streamed` remains the RT-kernel-gated
  alternative if the machine ever gets PREEMPT_RT + chrt.

## 6. Other fallbacks (if impedance disappoints)

1. **Streamed position under RT** — needs the rt_jitter verdict + chrt +
   ideally a PREEMPT_RT kernel (see 4b).
2. **Carrot-lead on async** (client-only, ~10 lines): keep the streamed target
   ahead of the arm by its stopping distance so the goal-seeker stays in its
   cruise phase. A workaround: corners get shaved, motion ends still park.
3. **Compiled helper**: a ~100-line C++ (or Cython) process that does
   readOnce/tracker/writeOnce and takes targets over a socket — the full ROS
   answer, more build effort.

## 7. Status ledger

- [x] streamed position controller written + wired (`--controller streamed`) — parked pending RT
- [x] GIL/GC/priority mitigations in server
- [x] jitter measurement tool (`tools/rt_jitter.py`)
- [x] policy-free acceptance probe (perception PC: `scripts/test/stream_probe.py`)
- [x] RealtimeConfig verified from source; decision: impedance executor (§5)
- [x] native zmq acceptance probe (`tools/impedance_probe.py`: hold/sine vs impedance_server,
      grades lag-compensated pk-pk residual vs 20 mrad); Kd-coefficient contract fixed in
      the drive_impedance/demo callers (the action's Kd slot is a coefficient on sqrt(Kp))
- [x] impedance_server smoke: held at home across multiple sessions, no reflex (2026-08-08)
- [x] sine probe passes on impedance: 11.7 mrad (j4, Kp 200) / 15.9 mrad (j7, Kp 50) lag-comp
      pk-pk, lag ~200 ms, no ringing at any gain — with Kd-coeff 0.7 + `--interp-time 0.1`
      (now the defaults). Old low gains fail on friction deadband + overdamping, not streaming.
- [ ] client `set_action` path (sender computes dq, fixed gains) behind a config switch   ← **you are here**
- [ ] close_drawer end-to-end on impedance executor
