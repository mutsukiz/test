"""
Phase 2 check (no robot, no teleimager): Quest -> Windows -> G1_29 arm IK -> first-person view in the Quest.

Place this file in  xr_teleoperate/teleop/  and run it from that directory:
    cd xr_teleoperate\\teleop
    pip install mujoco
    python ik_preview.py --cert televuer\\cert.pem --key televuer\\key.pem

- Quest browser : https://<Windows IP>:8012/?ws=wss://<Windows IP>:8012  -> "Virtual Reality"
- In the headset you see what the G1 head camera (D435 position on torso_link) would see,
  rendered with MuJoCo from assets/g1/g1_body29_hand14.xml, with the arms driven by the IK result.
  A table and a few objects are added in front of the robot as a reference.

Options:
    --view fpv|pattern   first-person robot view (default) or the old test pattern
    --meshcat            also show the meshcat visualizer (slower)
    --no-timer-fix       do NOT raise the Windows timer resolution (to compare)

Reported every second: loop rate, IK / render time (mean / max), loop overruns.
At startup: accuracy of time.sleep(4 ms) (robot_arm.py runs a ~250 Hz sleep loop).
"""
import argparse
import os
import sys
import time

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)

ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # same order as the 14-dof reduced model of G1_29_ArmIK

# objects placed in front of the robot (world frame, pelvis at x=y=0, z=0.793)
SCENE_EXTRA = """
    <body name="table" pos="0.55 0 0.70">
      <geom type="box" size="0.25 0.45 0.02" rgba="0.55 0.42 0.30 1"/>
      <geom type="box" pos="0.2 0.4 -0.35" size="0.02 0.02 0.35" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="0.2 -0.4 -0.35" size="0.02 0.02 0.35" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="-0.2 0.4 -0.35" size="0.02 0.02 0.35" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="-0.2 -0.4 -0.35" size="0.02 0.02 0.35" rgba="0.3 0.3 0.3 1"/>
    </body>
    <geom type="box" pos="0.42 0.15 0.745" size="0.025 0.025 0.025" rgba="0.9 0.1 0.1 1"/>
    <geom type="cylinder" pos="0.48 -0.15 0.78" size="0.03 0.06" rgba="0.1 0.3 0.9 1"/>
    <geom type="sphere" pos="0.60 0.0 0.76" size="0.04" rgba="0.1 0.8 0.2 1"/>
"""


class G1FirstPersonView:
    """Kinematic G1 rendered from the head camera (d435_link in the URDF)."""
    # d435_joint in g1_body29_hand14.urdf: parent torso_link, xyz / pitch
    CAM_POS = np.array([0.0576235, 0.01753, 0.42987])
    CAM_PITCH = 0.8307767239493009

    def __init__(self, assets_dir, h, w, fovy=58.0):
        import mujoco
        self.mj = mujoco
        xml_path = os.path.join(assets_dir, "g1_body29_hand14.xml")
        xml = open(xml_path, encoding="utf-8").read()
        meshdir = os.path.join(assets_dir, "meshes").replace("\\", "/")
        xml = xml.replace('meshdir="meshes"', f'meshdir="{meshdir}"')
        i = xml.rfind("</worldbody>")
        xml = xml[:i] + SCENE_EXTRA + xml[i:]
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.m.vis.global_.fovy = fovy
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, w)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, h)
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.renderer = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.set_arm_q(np.zeros(14))

    def set_arm_q(self, q):
        self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)

    def render_bgr(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        p = self.d.xpos[self.torso] + R @ self.CAM_POS
        c, s = np.cos(self.CAM_PITCH), np.sin(self.CAM_PITCH)
        fwd = R @ np.array([c, 0.0, -s])
        self.cam.distance = 1.0
        self.cam.lookat[:] = p + fwd * self.cam.distance
        self.cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0]))
        self.cam.elevation = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
        self.renderer.update_scene(self.d, camera=self.cam)
        rgb = self.renderer.render()
        return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR (render_to_xr expects BGR)


def windows_timer(enable):
    if os.name != "nt" or not enable:
        return None
    import ctypes
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
    return winmm


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
    img = np.full((h, w, 3), 40, np.uint8)
    x = (i * 8) % w
    img[:, max(0, x - 10):x + 10] = (0, 200, 255)
    cv2.putText(img, f"IK preview  frame {i}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["hand", "controller"], default="hand")
    ap.add_argument("--display", choices=["immersive", "ego", "pass-through"], default="immersive")
    ap.add_argument("--view", choices=["fpv", "pattern"], default="fpv")
    ap.add_argument("--frequency", type=float, default=30.0)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--meshcat", action="store_true", help="also show meshcat (slower)")
    ap.add_argument("--no-timer-fix", action="store_true", help="keep the default ~15.6 ms Windows timer")
    args = ap.parse_args()

    winmm = windows_timer(not args.no_timer_fix)
    m, p95, mx = measure_sleep()
    print(f"[timer] fix={'on' if winmm else 'off'}: time.sleep(4 ms) -> mean {m:.2f} ms, p95 {p95:.2f} ms,"
          f" max {mx:.2f} ms  => a 250 Hz loop would run at ~{1000 / m:.0f} Hz")

    from televuer import TeleVuerWrapper
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    H, W = 480, 640
    fpv = None
    if args.view == "fpv" and args.display != "pass-through":
        fpv = G1FirstPersonView(os.path.join(parent_dir, "assets", "g1"), H, W)
        print("[fpv] MuJoCo first-person renderer ready.")

    tv = TeleVuerWrapper(use_hand_tracking=(args.input == "hand"), binocular=False, img_shape=(H, W),
                         display_fps=args.frequency, display_mode=args.display,
                         zmq=(args.display != "pass-through"), webrtc=False,
                         cert_file=args.cert, key_file=args.key, arm_reference_mode="head_yaw")

    print("[ik] loading G1_29 model (first time may take a while)...")
    arm_ik = G1_29_ArmIK(Visualization=args.meshcat)
    print("[ik] ready. Open the Quest page and enter VR. Ctrl+C to stop.")

    period = 1.0 / args.frequency
    i = 0
    ik_t, rd_t, loops, overruns = [], [], 0, 0
    t_report = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            tele = tv.get_tele_data()
            if tele.motion_data_ready:
                t1 = time.perf_counter()
                sol_q, _ = arm_ik.solve_ik(tele.left_wrist_pose, tele.right_wrist_pose)
                ik_t.append((time.perf_counter() - t1) * 1000)
                if fpv is not None:
                    fpv.set_arm_q(sol_q)

            if args.display != "pass-through":
                t1 = time.perf_counter()
                img = fpv.render_bgr() if fpv is not None else make_pattern(H, W, i)
                tv.render_to_xr(img)
                rd_t.append((time.perf_counter() - t1) * 1000)
            i += 1
            loops += 1

            now = time.perf_counter()
            if now - t_report >= 1.0:
                rd = f"render {np.mean(rd_t):5.1f} ms" if rd_t else "render  -"
                if ik_t:
                    print(f"loop {loops / (now - t_report):5.1f} Hz | IK mean {np.mean(ik_t):6.2f} ms"
                          f"  max {np.max(ik_t):6.2f} ms | {rd} | overruns {overruns}")
                else:
                    print(f"loop {loops / (now - t_report):5.1f} Hz | {rd} | waiting for XR motion data...")
                ik_t, rd_t, loops, overruns = [], [], 0, 0
                t_report = now

            dt = time.perf_counter() - t0
            if dt > period:
                overruns += 1
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        pass
    finally:
        tv.close()
        if winmm:
            winmm.timeEndPeriod(1)
        print("closed.")


if __name__ == "__main__":  # required on Windows (multiprocessing uses "spawn")
    main()
