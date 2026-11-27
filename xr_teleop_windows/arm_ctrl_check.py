"""
Drive a (mock) G1 with xr_teleoperate's real G1_29_ArmController - no XR device needed.

Place in xr_teleoperate/teleop/ (next to teleop_hand_and_arm.py) and run from there:
    python arm_ctrl_check.py                 # debug mode  -> rt/lowcmd
    python arm_ctrl_check.py --motion        # motion mode -> rt/arm_sdk
    python arm_ctrl_check.py --iface <name>  # choose the network interface for DDS
    python arm_ctrl_check.py --no-timer-fix  # Windows: keep the default 15.6 ms timer (to compare)

Both arms swing slowly and the left/right Dex3 hands open and close.
Watch the g1_mock_robot.py log: lowcmd (or arm_sdk) rate should be ~250 Hz, crc_err 0,
mode_machine_mismatch 0, and the arm tracking error small.
"""
import argparse
import os
import sys
import time

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(this_dir))

LEFT_CLOSED = [0.0, 0.6, 1.2, -1.3, -1.4, -1.3, -1.4]
RIGHT_CLOSED = [0.0, -0.6, -1.2, 1.3, 1.4, 1.3, 1.4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None)
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--no-timer-fix", action="store_true")
    args = ap.parse_args()

    if sys.platform == "win32" and not args.no_timer_fix:
        import ctypes
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
        print("[check] Windows timer resolution set to 1 ms")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)
    from teleop.robot_control.robot_arm import G1_29_ArmController

    arm = G1_29_ArmController(motion_mode=args.motion)
    pubs, msgs = [], []
    for side in ("left", "right"):
        p = ChannelPublisher(f"rt/dex3/{side}/cmd", HandCmd_); p.Init()
        msg = unitree_hg_msg_dds__HandCmd_()
        for i in range(7):
            msg.motor_cmd[i].mode, msg.motor_cmd[i].kp, msg.motor_cmd[i].kd = 1, 1.5, 0.2
        pubs.append(p); msgs.append(msg)

    print(f"[check] driving arms ({'rt/arm_sdk' if args.motion else 'rt/lowcmd'}) and Dex3 for {args.seconds:.0f} s")
    t0 = time.perf_counter()
    t_rep = t0
    try:
        while time.perf_counter() - t0 < args.seconds:
            t = time.perf_counter() - t0
            s = 0.5 * (1 - np.cos(2 * np.pi * t / 4.0))  # 0 -> 1 -> 0 every 4 s
            target = np.zeros(14)
            target[[0, 7]] = -0.8 * s       # shoulder pitch (forward)
            target[[3, 10]] = 0.2 + 0.9 * s  # elbow
            arm.ctrl_dual_arm(target, np.zeros(14))
            for msg, closed, p in zip(msgs, (LEFT_CLOSED, RIGHT_CLOSED), pubs):
                for i in range(7):
                    msg.motor_cmd[i].q = closed[i] * s
                p.Write(msg)
            if time.perf_counter() - t_rep > 1.0:
                q = arm.get_current_dual_arm_q()
                print(f"[check] t={t:5.1f}s  L shoulder/elbow target {target[0]:+.2f}/{target[3]:+.2f}"
                      f"  actual {q[0]:+.2f}/{q[3]:+.2f}")
                t_rep = time.perf_counter()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    print("[check] going home...")
    arm.ctrl_dual_arm_go_home()
    print("[check] done.")


if __name__ == "__main__":
    main()
