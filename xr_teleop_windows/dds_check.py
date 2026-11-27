#!/usr/bin/env python3
"""
DDS round-trip check between two machines (Phase 3), using unitree_sdk2_python / CycloneDDS.

  robot side (Jetson):   python3 dds_check.py echo --iface <ethernet interface>
  host side (Windows):   python dds_check.py ping --iface <adapter name or leave empty>

"ping" sends a unitree_hg LowState_-sized message (the same size as rt/lowstate) at --hz,
"echo" sends it straight back. ping reports rate, loss and round-trip time (RTT).

Topics used: rt/xrcheck/ping and rt/xrcheck/pong (they do not interfere with a robot).
"""
import argparse
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_


def init(args):
    if args.iface:
        ChannelFactoryInitialize(args.domain, args.iface)
    else:
        ChannelFactoryInitialize(args.domain)


def run_echo(args):
    init(args)
    pub = ChannelPublisher("rt/xrcheck/pong", LowState_); pub.Init()
    count = [0]

    def handler(msg):
        pub.Write(msg)
        count[0] += 1

    sub = ChannelSubscriber("rt/xrcheck/ping", LowState_); sub.Init(handler, 10)
    print(f"[echo] waiting for pings on domain {args.domain}, iface {args.iface or 'default'} (Ctrl+C to stop)")
    last = 0
    try:
        while True:
            time.sleep(2.0)
            n = count[0]
            print(f"[echo] {(n - last) / 2.0:6.1f} msg/s echoed (total {n})")
            last = n
    except KeyboardInterrupt:
        pass


def run_ping(args):
    if sys.platform == "win32":
        import ctypes
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
    init(args)
    pub = ChannelPublisher("rt/xrcheck/ping", LowState_); pub.Init()
    sent = {}
    rtts, lock = [], threading.Lock()

    def handler(msg):
        t = time.perf_counter()
        with lock:
            t0 = sent.pop(msg.tick, None)
            if t0 is not None:
                rtts.append((t - t0) * 1000)

    sub = ChannelSubscriber("rt/xrcheck/pong", LowState_); sub.Init(handler, 10)
    msg = unitree_hg_msg_dds__LowState_()
    print(f"[ping] sending {args.hz:.0f} msg/s for {args.seconds:.0f} s on domain {args.domain}, "
          f"iface {args.iface or 'default'} ...")
    period = 1.0 / args.hz
    seq, n_sent = 0, 0
    t_end = time.perf_counter() + args.seconds
    t_rep = time.perf_counter()
    t_next = time.perf_counter()
    while time.perf_counter() < t_end:
        seq = (seq + 1) & 0xFFFFFFFF
        msg.tick = seq
        with lock:
            sent[seq] = time.perf_counter()
        pub.Write(msg)
        n_sent += 1
        now = time.perf_counter()
        if now - t_rep >= 1.0:
            with lock:
                r = np.array(rtts); rtts.clear()
                stale = [k for k, v in sent.items() if now - v > 1.0]
                for k in stale:
                    sent.pop(k)
            if len(r):
                print(f"[ping] sent {n_sent:4d}/s  received {len(r):4d}/s  lost {len(stale):4d}  "
                      f"RTT mean {r.mean():6.2f} ms  p95 {np.percentile(r, 95):6.2f} ms  max {r.max():6.2f} ms")
            else:
                print(f"[ping] sent {n_sent:4d}/s  received    0/s  -> no reply. Is 'echo' running? "
                      f"Same domain? Firewall? Right --iface?")
            n_sent, t_rep = 0, now
        t_next += period
        time.sleep(max(0.0, t_next - time.perf_counter()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["ping", "echo"])
    ap.add_argument("--iface", default=None, help="network interface name for CycloneDDS")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--hz", type=float, default=250.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    run_ping(args) if args.role == "ping" else run_echo(args)


if __name__ == "__main__":
    main()
