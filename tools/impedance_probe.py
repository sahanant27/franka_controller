#!/usr/bin/env python3
"""Policy-free acceptance probe for impedance_server.py (zmq client).

The impedance-path analog of the perception PC's stream_probe.py: streams a
--hz (10 Hz = policy rate) target — hold or sine — as 21-D set_action commands
[Δq(7), Kp(7), Kd-coeff(7)] with Δq = target - q_measured, samples get_state at
~50 Hz, and grades tracking at the end. This target→action mapping is the
reference implementation for the real policy client.

Verdict (sine): pk-pk residual after removing pure phase lag, vs --tolerance
(default 20 mrad, the STREAMED_CONTROL.md criterion). Raw residual and the
estimated lag are printed separately so smooth-but-lagging (tune interp_time /
Kp) is distinguishable from ringing/stutter (what the async controller shows on
this same test). Hold mode grades raw pk-pk wander across all joints.

Run impedance_server.py first, then (robot PC/localhost first to de-risk, later
from the perception PC with --addr tcp://10.42.0.1:5556):
  python tools/impedance_probe.py --mode hold --duration 10
  python tools/impedance_probe.py --mode sine --joint 6 --amp 0.15 --freq 0.2
"""
import argparse
import math
import time

import numpy as np
import zmq

# Mirrors DEFAULT_KP in controllers/joint_impedance_controller.py (not imported:
# that module needs pylibfranka, and this probe must also run on the perception PC).
DEFAULT_KP = [80.0, 80.0, 80.0, 80.0, 30.0, 20.0, 12.0]


def request(sock, msg, timeout_ms=2000):
    sock.send_json(msg)
    if sock.poll(timeout_ms) == 0:
        raise TimeoutError(f"no reply to {msg.get('cmd')!r}")
    rep = sock.recv_json()
    if not rep.get("ok"):
        raise RuntimeError(rep.get("error", "server error"))
    return rep


def analyze_sine(t, target, q, max_lag=1.5):
    """Return (raw_pkpk, lag_sec, lagcomp_pkpk) of the residual q(t) - target(t-lag).

    The lag is the time shift of the commanded signal that minimizes the RMS
    residual — it absorbs the structural delay of the interp ramp + PD tracking,
    leaving ringing/stutter/amplitude loss in lagcomp_pkpk.
    """
    t, target, q = np.asarray(t), np.asarray(target), np.asarray(q)
    raw = q - target
    best_lag, best_rms = 0.0, np.inf
    for lag in np.arange(0.0, max_lag, 0.005):
        shifted = np.interp(t - lag, t, target)
        m = t - lag >= t[0]                     # ignore samples before data starts
        rms = float(np.sqrt(np.mean((q[m] - shifted[m]) ** 2)))
        if rms < best_rms:
            best_lag, best_rms = lag, rms
    shifted = np.interp(t - best_lag, t, target)
    m = t - best_lag >= t[0]
    res = q[m] - shifted[m]
    return float(np.ptp(raw)), best_lag, float(np.ptp(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--addr", default="tcp://127.0.0.1:5556")
    ap.add_argument("--mode", choices=["hold", "sine"], default="sine")
    ap.add_argument("--joint", type=int, default=6, help="joint to oscillate (sine mode)")
    ap.add_argument("--amp", type=float, default=0.15, help="oscillation amplitude [rad]")
    ap.add_argument("--freq", type=float, default=0.2, help="oscillation frequency [Hz]")
    ap.add_argument("--hz", type=float, default=10.0, help="ACTION rate (policy rate)")
    ap.add_argument("--sample-hz", type=float, default=50.0, help="get_state sampling rate")
    ap.add_argument("--duration", type=float, default=15.0, help="run time [s]")
    ap.add_argument("--stiffness", type=float, default=None,
                    help="uniform Kp for all joints (default: per-joint DEFAULT_KP)")
    ap.add_argument("--kd-coeff", type=float, default=2.0,
                    help="Kd coefficient on sqrt(Kp) (policy range 0.3-2.0; 2.0 = critically damped)")
    ap.add_argument("--tolerance", type=float, default=0.020,
                    help="PASS threshold on the pk-pk residual [rad]")
    ap.add_argument("--csv", default=None, help="dump t,target,q of the probed joint to this file")
    args = ap.parse_args()

    kp = np.full(7, args.stiffness) if args.stiffness is not None else np.array(DEFAULT_KP)
    kd = np.full(7, args.kd_coeff)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.addr)

    print("ping ->", request(sock, {"cmd": "ping"}))
    q_start = np.array(request(sock, {"cmd": "get_state"})["q"])
    print(f"start q = {q_start.round(3).tolist()}  mode={args.mode}  "
          f"action {args.hz:g} Hz / sample {args.sample_hz:g} Hz")

    sample_dt = 1.0 / args.sample_hz
    ticks_per_action = max(1, round(args.sample_hz / args.hz))
    log_t, log_target, log_q = [], [], []
    t0 = time.monotonic()
    tick = 0
    try:
        while True:
            loop = time.monotonic()
            t = loop - t0
            if t >= args.duration:
                break

            rep = request(sock, {"cmd": "get_state"})
            if not rep.get("controlling", False):
                raise RuntimeError("server not controlling — 1 kHz loop is down")
            q = np.array(rep["q"])

            target = q_start.copy()
            if args.mode == "sine":
                target[args.joint] += args.amp * math.sin(2 * math.pi * args.freq * t)

            if tick % ticks_per_action == 0:
                dq = target - q                    # absolute tracking via delta-from-measured
                action = np.concatenate([dq, kp, kd])
                request(sock, {"cmd": "set_action", "a": action.tolist()})

            log_t.append(t)
            log_target.append(target[args.joint])
            log_q.append(q[args.joint])
            if tick % int(args.sample_hz) == 0:    # ~1 Hz progress line
                print(f"  t={t:5.1f}s  q[{args.joint}]={q[args.joint]:+.3f}  "
                      f"target={target[args.joint]:+.3f}  err={q[args.joint]-target[args.joint]:+.4f}")
            tick += 1
            sleep = sample_dt - (time.monotonic() - loop)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        # park the reference where the arm is so it doesn't keep chasing the last sine point
        q = np.array(request(sock, {"cmd": "get_state"})["q"])
        request(sock, {"cmd": "set_action", "a": np.concatenate([np.zeros(7), kp, kd]).tolist()})
        sock.close(0)
        ctx.term()

    if args.csv:
        np.savetxt(args.csv, np.column_stack([log_t, log_target, log_q]),
                   delimiter=",", header="t,target,q", comments="")
        print(f"wrote {args.csv} ({len(log_t)} samples)")

    print()
    if args.mode == "hold":
        pkpk = float(np.ptp(np.asarray(log_q)))
        print(f"hold wander on joint {args.joint}: pk-pk {pkpk*1e3:.1f} mrad "
              f"(final drift {abs(log_q[-1]-log_q[0])*1e3:.1f} mrad)")
        verdict = pkpk < args.tolerance
    else:
        raw_pkpk, lag, lc_pkpk = analyze_sine(log_t, log_target, log_q)
        print(f"sine joint {args.joint}, amp {args.amp:g} rad @ {args.freq:g} Hz:")
        print(f"  raw residual        : {raw_pkpk*1e3:6.1f} mrad pk-pk")
        print(f"  estimated lag       : {lag*1e3:6.0f} ms   (interp ramp + PD delay — tune, not a failure)")
        print(f"  lag-comp residual   : {lc_pkpk*1e3:6.1f} mrad pk-pk   <-- graded vs {args.tolerance*1e3:.0f} mrad")
        verdict = lc_pkpk < args.tolerance
    print("PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
