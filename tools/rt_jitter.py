#!/usr/bin/env python3
"""Measure whether THIS machine + interpreter can hold a 1 kHz loop (no robot).

Runs a 1 kHz sleep-locked loop for --secs with a competing busy thread
(simulating the zmq REP load) and reports how often an iteration overruns the
1 ms budget. Run it exactly like the server would run:

    python tools/rt_jitter.py
    chrt -f 80 python tools/rt_jitter.py      # under SCHED_FIFO, if permitted

Verdict guide: overruns >2 ms should be ~zero for a viable 1 kHz control loop.
Also prints kernel + RT permissions so we know what we're working with.
"""
import argparse
import gc
import os
import sys
import threading
import time


def busy_thread(stop):
    """Simulates the REP/gripper threads: bursts of allocation + json work."""
    import json
    while not stop.is_set():
        json.dumps({"q": [0.1] * 7, "ee_pose": [[0.0] * 4] * 4})
        time.sleep(0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=10.0)
    args = ap.parse_args()

    print(f"kernel   : {os.uname().release}")
    print(f"PREEMPT_RT: {'yes' if 'rt' in os.uname().release.lower() else 'no (or not in name)'}")
    try:
        sched = os.sched_getscheduler(0)
        print(f"scheduler: {'SCHED_FIFO/RR (RT)' if sched in (1, 2) else 'SCHED_OTHER (normal)'}")
    except AttributeError:
        pass

    sys.setswitchinterval(0.0005)
    gc.freeze()
    gc.disable()
    stop = threading.Event()
    threading.Thread(target=busy_thread, args=(stop,), daemon=True).start()

    n = int(args.secs * 1000)
    lat = []
    next_t = time.perf_counter() + 0.001
    for _ in range(n):
        while time.perf_counter() < next_t:      # spin the last stretch like an RT loop
            pass
        lat.append(time.perf_counter() - next_t)
        next_t += 0.001
    stop.set()

    lat_ms = sorted(x * 1000 for x in lat)
    def pct(p):
        return lat_ms[min(len(lat_ms) - 1, int(p / 100 * len(lat_ms)))]
    over1 = sum(1 for x in lat_ms if x > 1.0)
    over2 = sum(1 for x in lat_ms if x > 2.0)
    print(f"\n{n} iterations: overrun p50 {pct(50):.3f} ms  p99 {pct(99):.3f} ms  "
          f"p99.9 {pct(99.9):.3f} ms  max {lat_ms[-1]:.3f} ms")
    print(f">1 ms late: {over1} ({100 * over1 / n:.2f}%)   >2 ms late: {over2}")
    print("\nVERDICT: " + ("this machine can plausibly hold a Python 1 kHz loop"
                           if over2 == 0 and over1 < n * 0.001 else
                           "Python 1 kHz is NOT reliable here — use chrt -f, or the "
                           "compiled/async executor with lead targets, or impedance"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
