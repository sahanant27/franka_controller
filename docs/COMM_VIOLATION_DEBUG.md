# Debugging `communication_constraints_violation` (robot PC)

The firmware kills the session when too many 1 ms command cycles are missed.
Every occurrence has one of a small set of causes. Work this top to bottom ON
THE ROBOT PC.

## 0. First: WHEN did it fire? That names the suspect

| moment it fired | prime suspect | section |
|---|---|---|
| at server start / arming | gripper child still connecting while arming | §2 |
| the moment a gripper open/close was commanded | per-command process spawn mid-session (fixed in code — pull) | §3 |
| random, seconds-to-minutes in, no gripper involved | SCHED_FIFO not granted → CFS latency tail | §1 |
| under other load (browser, rviz, builds) | same as above + CPU contention | §1, §5 |

## 1. Is SCHED_FIFO actually granted? (the #1 silent killer)

The 1 kHz loop asks for FIFO 80 itself and only WARNS if denied — look for
this line in the server output:

    joint_impedance: SCHED_FIFO denied (...) — running best-effort

Check permissions:

```bash
groups                    # must include: realtime
ulimit -r                 # must be >= 80 (rtprio limit)
cat /etc/security/limits.d/*realtime* 2>/dev/null
```

If missing (this is the standard libfranka setup step):

```bash
sudo groupadd -f realtime
sudo usermod -aG realtime $USER
sudo tee /etc/security/limits.d/99-realtime.conf <<'EOF'
@realtime soft rtprio 99
@realtime hard rtprio 99
@realtime soft memlock unlimited
@realtime hard memlock unlimited
EOF
# then LOG OUT AND BACK IN (group membership is per-session), re-check `groups`
```

Validate before touching the robot (no motion, 10 s):

```bash
python tools/rt_jitter.py            # should now show SCHED_FIFO achievable
chrt -f 80 python tools/rt_jitter.py # target: zero overruns > 2 ms
```

Measured reference on this machine (2026-08-08): CFS tail ~1 ms+ under load →
intermittent violations; FIFO 80 → max overrun 0.24 ms, zero misses.

## 2. Start-order rule (bisected 2026-08-08)

Arming the RT session while the gripper child process is still connecting
kills the session in ~0.5 s. The servers enforce: GripperService -> wait_ready
-> arm. If you see the `gripper poll not ready` warning at startup, the arm
was armed without a settled gripper — fix the gripper connection (cable, FCI
gripper enabled in Desk) rather than ignoring it.

## 3. Gripper commands mid-session (DIAGNOSIS + fix to implement here)

**Symptom signature:** server runs fine, holds pose, tracks actions — then the
reflex fires the moment (or within ~1 s of) a gripper open/close command.

**Mechanism:** `GripperService.command()` currently SPAWNS a fresh process per
open/close. A spawn boots a whole new interpreter: ~1 s of imports
(pylibfranka, numpy), disk I/O, CPU burst on some core, plus a brand-new
gripper TCP connect — all while the 1 kHz loop runs. This is the same
disturbance class as the bisected §2 failure (connect-during-arming kills the
session in ~0.5 s), just triggered at command time instead of start time. The
poll process is innocent — it started before arming, exactly to avoid this.

**Quick confirmation before changing code:** with the server holding pose,
send one command from another terminal and watch the timing correlation:

```bash
python scripts/test/gripper_ctl.py close     # reflex within ~1 s of this => confirmed
```

**The fix (implement in `controllers/gripper_service.py`):** ONE persistent
worker process, started before arming, owning poll AND commands on a single
lifelong gripper connection. After startup the control process never creates
another process or gripper connection. Design:

```python
def _worker(ip, poll_s, width_v, grasped_v, busy_v, running_v, cmd_q):
    import queue as _q
    import pylibfranka as franka
    g = franka.Gripper(ip)                      # ONE connection, for life
    while running_v.value:
        try:                                    # doubles as the poll pacing
            cmd = cmd_q.get(timeout=poll_s)
        except (_q.Empty, InterruptedError):
            cmd = None
        if cmd is not None:
            busy_v.value = 1
            try:                                # reuse the grasp/open logic from
                gripper_do(g, **cmd)            # home.py — refactor set_gripper into
            except Exception:                   # gripper_do(g, ...) that takes an
                pass                            # existing connection, keep set_gripper
            busy_v.value = 0                    # as a one-shot wrapper for startup/reset
        try:
            gs = g.read_once()
            width_v.value = float(gs.width)
            grasped_v.value = int(gs.is_grasped)
        except Exception:
            pass
```

Changes to `GripperService`:
- `__init__`: replace the poll process with this worker; add
  `busy_v = Value("i", 0)` and `cmd_q = ctx.Queue(maxsize=4)`.
- `command()`: no Process creation — `if busy_v.value: return False;
  cmd_q.put_nowait({"action":..., "width":..., "force":...}); return True`.
- `stop()`: `running_v.value = 0; worker.join(1.5)` (grasp may be in flight),
  terminate if still alive.
- `wait_ready()` unchanged — it now also proves the worker is up.
- `home.py`: extract `gripper_do(g, action, width, speed, force, do_homing)`
  from `set_gripper`'s body; `set_gripper(ip, ...)` becomes
  `return gripper_do(franka.Gripper(ip), ...)`.

**Verify (this is also §6 step 2):**

```bash
# terminal A: server up, arm holding
python servers/impedance_server.py --ip 172.16.0.2 --max-dq 0.2 --no-gripper
# terminal B (or perception PC): hammer the gripper while watching terminal A
python scripts/test/gripper_ctl.py close && sleep 3
python scripts/test/gripper_ctl.py open  && sleep 3
python scripts/test/gripper_ctl.py close
# PASS: fingers cycle, arm never twitches, server prints NO reflex.
```

## 4. Other in-process suspects (already mitigated — verify not reverted)

- `sys.setswitchinterval(0.0005)` + `gc.freeze(); gc.disable()` before arming
  (impedance_server). FIFO does NOT make these redundant: a FIFO thread still
  waits on the GIL (priority does not transfer), and a gen-2 GC pause runs in
  whichever thread triggers it. Three independent legs: FIFO (OS), switch
  interval (GIL), GC off (interpreter). All three were active for the
  zero-miss measurement.
- `load_model()` BEFORE `start_torque_control()` (cold-start fetch blows the
  first deadlines otherwise).
- `wait_until_still()` before arming (torque-discontinuity on handoff).

## 5. Environment levers (if §1-4 are clean and it still fires)

```bash
uname -v | grep -i preempt      # PREEMPT_RT kernel? without it, worst-case
                                # latency is unbounded (franka docs mandate RT)
# reduce competing load while testing: close browsers/IDEs, and/or pin:
taskset -c 2,3 python servers/impedance_server.py ...   # keep loop off core 0
# power management jitter:
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # want: performance
```

## 6. Acceptance sequence after ANY change

1. Server starts, arm homes, holds 60 s — no reflex.
2. §3 gripper hammer — no reflex, arm steady.
3. One policy run from the perception PC (close_drawer, known-good) — completes.
4. Only then: new experiments.
