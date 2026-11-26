"""
Phase 1 check: Quest <-> Windows host only (no teleimager server, no robot).

- Starts TeleVuer on https://<this PC>:8012 with a generated test pattern as the "robot camera" image.
- Prints head / wrist positions, pinch values and how often the XR data is updated.

Usage (inside the conda env):
    python test_televuer_local.py                      # hand tracking, immersive (test pattern)
    python test_televuer_local.py --input controller   # controllers
    python test_televuer_local.py --display pass-through
    python test_televuer_local.py --cert cert.pem --key key.pem

Then open on the Quest browser:
    https://<Windows IP>:8012/?ws=wss://<Windows IP>:8012
"""
import argparse
import time

import cv2
import numpy as np


def make_pattern(h, w, frame_idx):
    img = np.zeros((h, w, 3), np.uint8)
    # color bars
    colors = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
              (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0)]
    bw = w // len(colors)
    for i, c in enumerate(colors):
        img[: h * 2 // 3, i * bw:(i + 1) * bw] = c
    # moving bar shows that the image stream is alive
    x = (frame_idx * 8) % w
    img[h * 2 // 3:, max(0, x - 10):x + 10] = (0, 200, 255)
    cv2.putText(img, f"frame {frame_idx}  {time.strftime('%H:%M:%S')}", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img  # BGR (render_to_xr converts BGR->RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["hand", "controller"], default="hand")
    ap.add_argument("--display", choices=["immersive", "ego", "pass-through"], default="immersive")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()

    from televuer import TeleVuer

    H, W = 480, 640
    use_hand = args.input == "hand"
    tv = TeleVuer(use_hand_tracking=use_hand, binocular=False, img_shape=(H, W),
                  display_fps=args.fps, display_mode=args.display,
                  zmq=(args.display != "pass-through"), webrtc=False,
                  cert_file=args.cert, key_file=args.key)

    print("TeleVuer started. Open https://<Windows IP>:8012/?ws=wss://<Windows IP>:8012 on the Quest.")
    print("Ctrl+C to stop.")

    frame_idx = 0
    last_head = None
    updates = 0
    t_report = time.perf_counter()
    period = 1.0 / args.fps
    try:
        while True:
            t0 = time.perf_counter()
            if args.display != "pass-through":
                tv.render_to_xr(make_pattern(H, W, frame_idx))
            frame_idx += 1

            head = np.array(tv.head_pose)
            if last_head is None or not np.array_equal(head, last_head):
                updates += 1
                last_head = head.copy()

            now = time.perf_counter()
            if now - t_report >= 1.0:
                ready = bool(tv.motion_data_ready_shared.value)
                print("=" * 70)
                print(f"XR data update rate: {updates / (now - t_report):5.1f} Hz   motion_data_ready={ready}")
                print(f"head pos   : {np.round(head[:3, 3], 3)}")
                print(f"left  arm  : {np.round(np.array(tv.left_arm_pose)[:3, 3], 3)}")
                print(f"right arm  : {np.round(np.array(tv.right_arm_pose)[:3, 3], 3)}")
                if use_hand:
                    print(f"pinch L/R  : {tv.left_hand_pinchValue:.2f} / {tv.right_hand_pinchValue:.2f}")
                else:
                    print(f"trigger L/R: {tv.left_ctrl_triggerValue:.2f} / {tv.right_ctrl_triggerValue:.2f}"
                          f"   stick L={np.round(tv.left_ctrl_thumbstickValue, 2)}")
                updates = 0
                t_report = now

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        tv.close()
        print("closed.")


if __name__ == "__main__":  # required on Windows (multiprocessing uses "spawn")
    main()
