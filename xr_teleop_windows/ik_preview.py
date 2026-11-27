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
    --fpv-size 320x240   render the first-person view at a lower resolution
    --fpv-fps 30         target frame rate of the first-person view (rendered in its own thread)
    --fpv-hq             shadows / reflections / anti-aliasing on (slower)
    --stereo             render a left and a right eye view (depth perception in the headset; 2x render cost)

Reported every second: IK loop rate, IK time (mean / max), loop overruns, first-person render time / fps.
At startup: accuracy of time.sleep(4 ms) (robot_arm.py runs a ~250 Hz sleep loop) and the OpenGL renderer.
"""
import argparse
import os
import sys
import threading
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

# Dex3 joints in the order of the Dex3 API / HandRetargeting (thumb 0-2, middle 0-1, index 0-1)
LEFT_HAND_JOINTS = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                    "left_hand_index_0_joint", "left_hand_index_1_joint"]
RIGHT_HAND_JOINTS = [n.replace("left_", "right_") for n in LEFT_HAND_JOINTS]
# "closed" pose for the simple open/close mode (signs follow the joint ranges of the MJCF)
LEFT_HAND_CLOSED = np.array([0.0, 0.6, 1.2, -1.3, -1.4, -1.3, -1.4])
RIGHT_HAND_CLOSED = np.array([0.0, -0.6, -1.2, 1.3, 1.4, 1.3, 1.4])

# objects placed in front of the robot (world frame, pelvis at x=y=0, z=0.793).
# G1 is small: shoulder at z~1.085, so the table top is at z=0.78 and its near edge at x=0.20.
SCENE_EXTRA = """
    <body name="table" pos="0.42 0 0.76">
      <geom type="box" size="0.22 0.40 0.02" rgba="0.55 0.42 0.30 1"/>
      <geom type="box" pos="0.18 0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="0.18 -0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="-0.18 0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="-0.18 -0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
    </body>
    <geom type="box" pos="0.30 0.15 0.805" size="0.025 0.025 0.025" rgba="0.9 0.1 0.1 1"/>
    <geom type="cylinder" pos="0.33 -0.15 0.84" size="0.03 0.06" rgba="0.1 0.3 0.9 1"/>
    <geom type="sphere" pos="0.40 0.0 0.82" size="0.04" rgba="0.1 0.8 0.2 1"/>
"""


class G1FirstPersonView:
    """Kinematic G1 rendered from the head camera (d435_link in the URDF)."""
    # d435_joint in g1_body29_hand14.urdf: parent torso_link, xyz / pitch
    CAM_POS = np.array([0.0576235, 0.01753, 0.42987])
    CAM_PITCH = 0.8307767239493009

    def __init__(self, assets_dir, h, w, fovy=58.0, high_quality=False):
        import mujoco
        self.mj = mujoco
        xml_path = os.path.join(assets_dir, "g1_body29_hand14.xml")
        xml = open(xml_path, encoding="utf-8").read()
        meshdir = os.path.join(assets_dir, "meshes").replace("\\", "/")
        xml = xml.replace('meshdir="meshes"', f'meshdir="{meshdir}"')
        # extra objects and the floor go to geom group 2, so that group 0 (= duplicated
        # collision meshes of the robot) can be hidden without hiding them
        xml = xml.replace('<geom name="floor"', '<geom name="floor" group="2"')
        i = xml.rfind("</worldbody>")
        xml = xml[:i] + SCENE_EXTRA.replace("<geom ", '<geom group="2" ') + xml[i:]
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.m.vis.global_.fovy = fovy
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, w)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, h)
        if not high_quality:
            self.m.vis.quality.offsamples = 0
        self.d = mujoco.MjData(self.m)
        adr = lambda names: [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        self.qadr = adr(ARM_JOINTS)
        self.lhadr, self.rhadr = adr(LEFT_HAND_JOINTS), adr(RIGHT_HAND_JOINTS)
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.renderer = mujoco.Renderer(self.m, h, w)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0  # hide collision meshes (each robot mesh exists twice)
        if not high_quality:
            for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                      mujoco.mjtRndFlag.mjRND_SKYBOX):
                self.renderer.scene.flags[f] = False
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.set_arm_q(np.zeros(14))

    def gl_info(self):
        try:
            self.renderer._gl_context.make_current()
            from OpenGL import GL
            return (f"{GL.glGetString(GL.GL_VENDOR).decode()} / {GL.glGetString(GL.GL_RENDERER).decode()}"
                    f" / OpenGL {GL.glGetString(GL.GL_VERSION).decode()}")
        except Exception as e:
            return f"unknown ({e})"

    def set_arm_q(self, q, left_hand_q=None, right_hand_q=None):
        self.d.qpos[self.qadr] = q
        if left_hand_q is not None:
            self.d.qpos[self.lhadr] = left_hand_q
        if right_hand_q is not None:
            self.d.qpos[self.rhadr] = right_hand_q
        self.mj.mj_kinematics(self.m, self.d)

    def render_bgr(self, eye_offset=0.0):
        """eye_offset: shift along the camera's left axis [m] (+ = left eye, - = right eye)."""
        R = self.d.xmat[self.torso].reshape(3, 3)
        p = self.d.xpos[self.torso] + R @ (self.CAM_POS + np.array([0.0, eye_offset, 0.0]))
        c, s = np.cos(self.CAM_PITCH), np.sin(self.CAM_PITCH)
        fwd = R @ np.array([c, 0.0, -s])
        self.cam.distance = 1.0
        self.cam.lookat[:] = p + fwd * self.cam.distance
        self.cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0]))
        self.cam.elevation = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
        self.renderer.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        rgb = self.renderer.render()
        return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR (render_to_xr expects BGR)


class FpvThread(threading.Thread):
    """Renders the first-person view in its own thread so a slow GPU does not slow down the IK loop.
    The MuJoCo GL context is created inside this thread (GL contexts are bound to a thread)."""

    def __init__(self, tv, assets_dir, h, w, fps, high_quality, stereo=False, ipd=0.064):
        super().__init__(daemon=True)
        self.stereo, self.ipd = stereo, ipd
        self.tv, self.assets_dir, self.h, self.w = tv, assets_dir, h, w
        self.period = 1.0 / fps
        self.high_quality = high_quality
        self.lock = threading.Lock()
        self.q = np.zeros(14)
        self.lh = np.zeros(7)
        self.rh = np.zeros(7)
        self.times = []
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None

    def set_arm_q(self, q, left_hand_q=None, right_hand_q=None):
        with self.lock:
            self.q = np.array(q, copy=True)
            if left_hand_q is not None:
                self.lh = np.array(left_hand_q, copy=True)
            if right_hand_q is not None:
                self.rh = np.array(right_hand_q, copy=True)

    def pop_times(self):
        with self.lock:
            t, self.times = self.times, []
        return t

    def run(self):
        try:
            fpv = G1FirstPersonView(self.assets_dir, self.h, self.w, high_quality=self.high_quality)
            print(f"[fpv] OpenGL: {fpv.gl_info()}")
        except Exception as e:
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            with self.lock:
                q, lh, rh = self.q.copy(), self.lh.copy(), self.rh.copy()
            fpv.set_arm_q(q, lh, rh)
            if self.stereo:  # side-by-side: left half = left eye (televuer binocular layout)
                img = np.hstack([fpv.render_bgr(+self.ipd / 2), fpv.render_bgr(-self.ipd / 2)])
            else:
                img = fpv.render_bgr()
            self.tv.render_to_xr(img)
            dt = time.perf_counter() - t0
            with self.lock:
                self.times.append(dt * 1000)
            time.sleep(max(0.0, self.period - dt))


class DexHand:
    """Dex3 finger targets from the XR data.
    - "retarget": the same dex-retargeting pipeline as robot_hand_unitree.py (hand tracking only)
    - "simple"  : open/close by pinch (hand tracking) or trigger (controller)"""

    def __init__(self, mode, use_hand_tracking):
        self.mode = "simple"
        self.retargeting = None
        if mode in ("auto", "retarget") and use_hand_tracking:
            try:
                from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
                self.retargeting = HandRetargeting(HandType.UNITREE_DEX3)
                self.mode = "retarget"
            except Exception as e:
                if mode == "retarget":
                    raise
                print(f"[hand] dex-retargeting not available ({type(e).__name__}: {e}) -> simple open/close")
        print(f"[hand] mode: {self.mode}")
        self.closure = (0.0, 0.0)

    @staticmethod
    def _closure_from_distance(v):
        # tv_wrapper: pinch / trigger values go ~10 (open / released) -> 0 (pinched / fully pressed)
        return float(np.clip(1.0 - v / 10.0, 0.0, 1.0))

    def compute(self, tele, use_hand_tracking):
        if self.mode == "retarget":
            r = self.retargeting
            lp, rp = np.asarray(tele.left_hand_pos), np.asarray(tele.right_hand_pos)
            ref_l = lp[r.left_indices[1, :]] - lp[r.left_indices[0, :]]
            ref_r = rp[r.right_indices[1, :]] - rp[r.right_indices[0, :]]
            lq = r.left_retargeting.retarget(ref_l)[r.left_dex_retargeting_to_hardware]
            rq = r.right_retargeting.retarget(ref_r)[r.right_dex_retargeting_to_hardware]
            return lq, rq
        if use_hand_tracking:
            cl = max(self._closure_from_distance(tele.left_hand_pinchValue), tele.left_hand_squeezeValue)
            cr = max(self._closure_from_distance(tele.right_hand_pinchValue), tele.right_hand_squeezeValue)
        else:
            cl = self._closure_from_distance(tele.left_ctrl_triggerValue)
            cr = self._closure_from_distance(tele.right_ctrl_triggerValue)
        self.closure = (cl, cr)
        return LEFT_HAND_CLOSED * cl, RIGHT_HAND_CLOSED * cr


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
    ap.add_argument("--fpv-size", default="640x480", help="WxH of the first-person view")
    ap.add_argument("--fpv-fps", type=float, default=30.0)
    ap.add_argument("--fpv-hq", action="store_true", help="shadows / reflections / anti-aliasing on")
    ap.add_argument("--hand", choices=["auto", "retarget", "simple", "off"], default="auto",
                    help="Dex3 fingers: dex-retargeting (hand tracking), simple open/close, or off")
    ap.add_argument("--stereo", action="store_true", help="render left/right eye views (depth perception)")
    ap.add_argument("--ipd", type=float, default=0.064, help="eye distance for --stereo [m]")
    args = ap.parse_args()

    winmm = windows_timer(not args.no_timer_fix)
    m, p95, mx = measure_sleep()
    print(f"[timer] fix={'on' if winmm else 'off'}: time.sleep(4 ms) -> mean {m:.2f} ms, p95 {p95:.2f} ms,"
          f" max {mx:.2f} ms  => a 250 Hz loop would run at ~{1000 / m:.0f} Hz")

    from televuer import TeleVuerWrapper
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    use_fpv = args.view == "fpv" and args.display != "pass-through"
    W, H = (int(v) for v in args.fpv_size.lower().split("x")) if use_fpv else (640, 480)
    stereo = use_fpv and args.stereo

    tv = TeleVuerWrapper(use_hand_tracking=(args.input == "hand"), binocular=stereo,
                         img_shape=(H, 2 * W if stereo else W),
                         display_fps=args.frequency, display_mode=args.display,
                         zmq=(args.display != "pass-through"), webrtc=False,
                         cert_file=args.cert, key_file=args.key, arm_reference_mode="head_yaw")

    fpv = None
    if use_fpv:
        fpv = FpvThread(tv, os.path.join(parent_dir, "assets", "g1"), H, W, args.fpv_fps, args.fpv_hq,
                        stereo=stereo, ipd=args.ipd)
        fpv.start()
        fpv.ready.wait()
        if fpv.error:
            raise fpv.error
        print(f"[fpv] MuJoCo first-person renderer ready ({W}x{H}{' per eye, stereo' if stereo else ''}).")

    use_hand_tracking = args.input == "hand"
    print(f"[xr] input mode: {args.input}"
          + ("  (use --input controller when holding the Quest controllers)" if use_hand_tracking else ""))
    hand = DexHand(args.hand, use_hand_tracking) if args.hand != "off" else None

    print("[ik] loading G1_29 model (first time may take a while)...")
    arm_ik = G1_29_ArmIK(Visualization=args.meshcat)
    print("[ik] ready. Open the Quest page and enter VR. Ctrl+C to stop.")

    period = 1.0 / args.frequency
    i = 0
    ik_t, hd_t, loops, overruns = [], [], 0, 0
    no_data_since = time.perf_counter()
    t_report = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            tele = tv.get_tele_data()
            if tele.motion_data_ready:
                t1 = time.perf_counter()
                sol_q, _ = arm_ik.solve_ik(tele.left_wrist_pose, tele.right_wrist_pose)
                ik_t.append((time.perf_counter() - t1) * 1000)
                lh = rh = None
                if hand is not None:
                    t1 = time.perf_counter()
                    lh, rh = hand.compute(tele, use_hand_tracking)
                    hd_t.append((time.perf_counter() - t1) * 1000)
                if fpv is not None:
                    fpv.set_arm_q(sol_q, lh, rh)

            if fpv is None and args.display != "pass-through":
                tv.render_to_xr(make_pattern(H, W, i))
            i += 1
            loops += 1

            now = time.perf_counter()
            if now - t_report >= 1.0:
                el = now - t_report
                rd = ""
                if fpv is not None:
                    rt = fpv.pop_times()
                    rd = (f" | fpv {len(rt) / el:4.1f} fps, render {np.mean(rt):5.1f} ms" if rt else " | fpv -")
                ik = (f"IK mean {np.mean(ik_t):6.2f} ms  max {np.max(ik_t):6.2f} ms" if ik_t
                      else "waiting for XR motion data...")
                hd = ""
                if hd_t:
                    hd = f" | hand {np.mean(hd_t):5.2f} ms"
                    if hand.mode == "simple":
                        hd += f" close L/R {hand.closure[0]:.2f}/{hand.closure[1]:.2f}"
                print(f"loop {loops / el:5.1f} Hz | {ik}{hd} | overruns {overruns}{rd}")
                if ik_t:
                    no_data_since = now
                elif now - no_data_since > 10:
                    print(f"[xr] no {'hand' if use_hand_tracking else 'controller'} data for 10 s. "
                          f"Holding controllers? -> --input controller. Using bare hands? -> --input hand")
                    no_data_since = now
                ik_t, hd_t, loops, overruns = [], [], 0, 0
                t_report = now

            dt = time.perf_counter() - t0
            if dt > period:
                overruns += 1
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        pass
    finally:
        if fpv is not None:
            fpv.stop_event.set()
            fpv.join(timeout=1.0)
        tv.close()
        if winmm:
            winmm.timeEndPeriod(1)
        print("closed.")


if __name__ == "__main__":  # required on Windows (multiprocessing uses "spawn")
    main()
