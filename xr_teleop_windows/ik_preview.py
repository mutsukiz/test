"""
Phase 2 check (no robot, no teleimager): Quest -> Windows -> G1_29 arm IK -> meshcat.

Place this file in  xr_teleoperate/teleop/  and run it from that directory:
    cd xr_teleoperate\\teleop
    python ik_preview.py --cert ..\\teleop\\televuer\\cert.pem --key ..\\teleop\\televuer\\key.pem

- Quest browser : https://<Windows IP>:8012/?ws=wss://<Windows IP>:8012
- meshcat (PC)  : opens automatically (http://127.0.0.1:7000/static/)

What it reports every second:
- loop rate, IK solve time (mean / max), number of loop overruns
At startup it also measures how accurate time.sleep(0.004) is on this PC
(the arm controller thread in robot_arm.py runs a ~250 Hz sleep loop).
"""
import argparse
import os
import sys
import time

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)


def measure_sleep(target=0.004, n=200):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        time.sleep(target)
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    return t.mean(), np.percentile(t, 95), t.max()


def make_pattern(h, w, i):
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = (40, 40, 40)
    x = (i * 8) % w
    img[:, max(0, x - 10):x + 10] = (0, 200, 255)
    cv2.putText(img, f"IK preview  frame {i}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["hand", "controller"], default="hand")
    ap.add_argument("--display", choices=["immersive", "ego", "pass-through"], default="immersive")
    ap.add_argument("--frequency", type=float, default=30.0)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--no-vis", action="store_true", help="disable meshcat (measure pure IK time)")
    args = ap.parse_args()

    m, p95, mx = measure_sleep()
    print(f"[timer] time.sleep(4 ms) -> mean {m:.2f} ms, p95 {p95:.2f} ms, max {mx:.2f} ms"
          f"  => effective ~{1000 / m:.0f} Hz for a 250 Hz loop")

    from televuer import TeleVuerWrapper
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    H, W = 480, 640
    tv = TeleVuerWrapper(use_hand_tracking=(args.input == "hand"), binocular=False, img_shape=(H, W),
                         display_fps=args.frequency, display_mode=args.display,
                         zmq=(args.display != "pass-through"), webrtc=False,
                         cert_file=args.cert, key_file=args.key, arm_reference_mode="head_yaw")

    print("[ik] loading G1_29 model (first time may take a while)...")
    arm_ik = G1_29_ArmIK(Visualization=not args.no_vis)
    print("[ik] ready. Open the Quest page and enter VR. Ctrl+C to stop.")

    period = 1.0 / args.frequency
    i = 0
    ik_times, loops, overruns = [], 0, 0
    t_report = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            if args.display != "pass-through":
                tv.render_to_xr(make_pattern(H, W, i))
            i += 1

            tele = tv.get_tele_data()
            if tele.motion_data_ready:
                t_ik = time.perf_counter()
                sol_q, sol_tauff = arm_ik.solve_ik(tele.left_wrist_pose, tele.right_wrist_pose)
                ik_times.append((time.perf_counter() - t_ik) * 1000)
            loops += 1

            now = time.perf_counter()
            if now - t_report >= 1.0:
                if ik_times:
                    print(f"loop {loops / (now - t_report):5.1f} Hz | IK mean {np.mean(ik_times):6.2f} ms"
                          f"  max {np.max(ik_times):6.2f} ms | overruns {overruns}"
                          f" | L wrist {np.round(tele.left_wrist_pose[:3, 3], 2)}"
                          f" R wrist {np.round(tele.right_wrist_pose[:3, 3], 2)}")
                else:
                    print(f"loop {loops / (now - t_report):5.1f} Hz | waiting for XR motion data...")
                ik_times, loops, overruns = [], 0, 0
                t_report = now

            dt = time.perf_counter() - t0
            if dt > period:
                overruns += 1
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        pass
    finally:
        tv.close()
        print("closed.")


if __name__ == "__main__":  # required on Windows (multiprocessing uses "spawn")
    main()
