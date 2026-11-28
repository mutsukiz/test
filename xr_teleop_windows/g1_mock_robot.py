#!/usr/bin/env python3
"""
G1 (29 dof) + Dex3 mock robot for testing xr_teleoperate without the real robot.

Runs a MuJoCo physics simulation of the G1 upper body (pelvis fixed in the air, table and
graspable objects in front) and speaks the same DDS topics as the real robot:

  subscribes  rt/lowcmd            (unitree_hg LowCmd_)   debug mode   (teleop without --motion)
              rt/arm_sdk           (unitree_hg LowCmd_)   motion mode  (teleop with --motion), weight = motor_cmd[29].q
              rt/dex3/left/cmd     (unitree_hg HandCmd_)
              rt/dex3/right/cmd    (unitree_hg HandCmd_)
  publishes   rt/lowstate          (unitree_hg LowState_)  500 Hz, with CRC
              rt/dex3/left/state   (unitree_hg HandState_) 100 Hz
              rt/dex3/right/state  (unitree_hg HandState_) 100 Hz

Each motor is simulated like the real joint controller:  tau = kp*(q_cmd - q) + kd*(dq_cmd - dq) + tau_cmd
using the kp / kd / q / dq / tau values that xr_teleoperate sends (clipped to the joint torque limits).
Without fresh commands the robot holds its default pose (like a locked / standing robot).

Every 2 s it prints what it received: command rates, CRC errors, mode_machine mismatches,
NaNs, gains, arm_sdk weight, tracking error and the real-time factor of the simulation.

Usage (Jetson, robot side):
    python3 g1_mock_robot.py --assets ~/xr_teleoperate/assets/g1 --iface <ethernet interface> --viewer
Usage (same PC as xr_teleoperate, for a local test):
    python g1_mock_robot.py --assets ..\\assets\\g1
"""
import argparse
import os
import re
import threading
import time

import numpy as np
import mujoco

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__HandState_
from unitree_sdk2py.utils.crc import CRC

# G1 29-dof motor order of LowCmd_/LowState_ (same as G1_29_JointIndex in xr_teleoperate)
BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ARM_IDS = list(range(15, 29))
WAIST_IDS = [12, 13, 14]
ARM_SDK_WEIGHT_ID = 29  # kNotUsedJoint0
# Dex3 motor order (thumb 0-2, middle 0-1, index 0-1)
LEFT_HAND_JOINTS = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                    "left_hand_index_0_joint", "left_hand_index_1_joint"]
RIGHT_HAND_JOINTS = [n.replace("left_", "right_") for n in LEFT_HAND_JOINTS]

# internal "hold" controller used when no (fresh) command is received
HOLD_KP = np.array([150.0] * 15 + [40.0] * 14)
HOLD_KD = np.array([5.0] * 15 + [2.0] * 14)
HAND_HOLD_KP, HAND_HOLD_KD = 1.5, 0.2
CMD_TIMEOUT = 0.2  # s

SCENE = """
    <body name="table" pos="0.42 0 0.76">
      <geom type="box" size="0.22 0.40 0.02" rgba="0.55 0.42 0.30 1" friction="1 0.01 0.001"/>
      <geom type="box" pos="0.18 0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="0.18 -0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="-0.18 0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
      <geom type="box" pos="-0.18 -0.36 -0.38" size="0.02 0.02 0.38" rgba="0.3 0.3 0.3 1"/>
    </body>
    <body name="red_cube" pos="0.30 0.15 0.81">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.08" rgba="0.9 0.1 0.1 1" friction="1.5 0.01 0.001"/>
    </body>
    <body name="blue_can" pos="0.33 -0.15 0.845">
      <freejoint/>
      <geom type="cylinder" size="0.03 0.06" mass="0.12" rgba="0.1 0.3 0.9 1" friction="1.5 0.01 0.001"/>
    </body>
    <body name="green_ball" pos="0.40 0.0 0.825">
      <freejoint/>
      <geom type="sphere" size="0.04" mass="0.05" rgba="0.1 0.8 0.2 1" friction="1.5 0.01 0.001"/>
    </body>
"""


def build_model(assets_dir, with_objects=True, timestep=0.002):
    xml = open(os.path.join(assets_dir, "g1_body29_hand14.xml"), encoding="utf-8").read()
    meshdir = os.path.join(assets_dir, "meshes").replace("\\", "/")
    xml = xml.replace('meshdir="meshes"', f'meshdir="{meshdir}"')
    # fix the pelvis in the air (no free joint) -> upper-body teleoperation test stand
    xml = re.sub(r'<joint name="floating_base_joint"[^>]*/>', "", xml)
    # implicit integration of the joint PD damping keeps light finger links stable
    xml = re.sub(r"(<compiler[^>]*/>)",
                 r'\1\n  <option timestep="%g" integrator="implicitfast"/>\n'
                 r'  <default><joint armature="0.01" damping="0.05"/></default>' % timestep, xml, count=1)
    # replace the plain motors by general actuators whose PD gains are set every step
    names = BODY_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
    acts = "\n".join(f'    <general name="{n}" joint="{n}" gaintype="fixed" biastype="affine" '
                     f'gainprm="0" biasprm="0 0 0"/>' for n in names)
    xml = re.sub(r"<actuator>.*?</actuator>", f"<actuator>\n{acts}\n  </actuator>", xml, flags=re.S)
    if with_objects:
        i = xml.rfind("</worldbody>")
        xml = xml[:i] + SCENE + xml[i:]
    return mujoco.MjModel.from_xml_string(xml), names


class Cmd:
    """Latest motor commands of one topic (thread-safe copy from the DDS callback)."""

    def __init__(self, n):
        self.q = np.zeros(n); self.dq = np.zeros(n); self.kp = np.zeros(n); self.kd = np.zeros(n)
        self.tau = np.zeros(n); self.mode = np.zeros(n, dtype=int)
        self.stamp = 0.0
        self.count = 0
        self.crc_err = 0
        self.mm_mismatch = 0
        self.nan = 0
        self.lock = threading.Lock()

    def fresh(self, now):
        return now - self.stamp < CMD_TIMEOUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="xr_teleoperate/assets/g1 directory")
    ap.add_argument("--iface", default=None, help="network interface for DDS (e.g. eth0, enP8p1s0)")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain (0 = real robot, as used without --sim)")
    ap.add_argument("--mode-machine", type=int, default=5, help="value published in LowState_.mode_machine")
    ap.add_argument("--state-hz", type=float, default=250.0,
                    help="rt/lowstate publish rate (real robot: 500 Hz; 250 Hz = xr_teleoperate's control rate)")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--viewer-fps", type=float, default=20.0,
                    help="viewer refresh rate; lower it if 'realtime' drops below 1.0 (slow GPU)")
    ap.add_argument("--no-objects", action="store_true")
    args = ap.parse_args()

    m, act_names = build_model(args.assets, with_objects=not args.no_objects)
    d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    body_q = np.array([m.jnt_qposadr[jid(n)] for n in BODY_JOINTS])
    body_v = np.array([m.jnt_dofadr[jid(n)] for n in BODY_JOINTS])
    lh_q = np.array([m.jnt_qposadr[jid(n)] for n in LEFT_HAND_JOINTS])
    lh_v = np.array([m.jnt_dofadr[jid(n)] for n in LEFT_HAND_JOINTS])
    rh_q = np.array([m.jnt_qposadr[jid(n)] for n in RIGHT_HAND_JOINTS])
    rh_v = np.array([m.jnt_dofadr[jid(n)] for n in RIGHT_HAND_JOINTS])
    n_body, n_hand = len(BODY_JOINTS), 7
    mujoco.mj_forward(m, d)

    ChannelFactoryInitialize(args.domain, args.iface) if args.iface else ChannelFactoryInitialize(args.domain)
    crc = CRC()

    lowcmd, armsdk = Cmd(35), Cmd(35)
    lhcmd, rhcmd = Cmd(n_hand), Cmd(n_hand)

    def on_lowcmd(store):
        def handler(msg: LowCmd_):
            bad_crc = crc.Crc(msg) != msg.crc
            with store.lock:
                for i in range(35):
                    c = msg.motor_cmd[i]
                    store.q[i], store.dq[i], store.kp[i], store.kd[i], store.tau[i], store.mode[i] = \
                        c.q, c.dq, c.kp, c.kd, c.tau, c.mode
                if not (np.all(np.isfinite(store.q)) and np.all(np.isfinite(store.tau))):
                    store.nan += 1
                    store.q[~np.isfinite(store.q)] = 0.0
                    store.tau[~np.isfinite(store.tau)] = 0.0
                store.crc_err += int(bad_crc)
                store.mm_mismatch += int(msg.mode_machine != args.mode_machine)
                store.count += 1
                store.stamp = time.perf_counter()
        return handler

    def on_handcmd(store):
        def handler(msg: HandCmd_):
            with store.lock:
                for i in range(n_hand):
                    c = msg.motor_cmd[i]
                    store.q[i], store.dq[i], store.kp[i], store.kd[i], store.tau[i], store.mode[i] = \
                        c.q, c.dq, c.kp, c.kd, c.tau, c.mode
                if not np.all(np.isfinite(store.q)):
                    store.nan += 1
                    store.q[~np.isfinite(store.q)] = 0.0
                store.count += 1
                store.stamp = time.perf_counter()
        return handler

    subs = []
    for topic, typ, store, mk in [("rt/lowcmd", LowCmd_, lowcmd, on_lowcmd), ("rt/arm_sdk", LowCmd_, armsdk, on_lowcmd),
                                  ("rt/dex3/left/cmd", HandCmd_, lhcmd, on_handcmd),
                                  ("rt/dex3/right/cmd", HandCmd_, rhcmd, on_handcmd)]:
        s = ChannelSubscriber(topic, typ)
        s.Init(mk(store), 10)
        subs.append(s)

    pub_low = ChannelPublisher("rt/lowstate", LowState_); pub_low.Init()
    pub_lh = ChannelPublisher("rt/dex3/left/state", HandState_); pub_lh.Init()
    pub_rh = ChannelPublisher("rt/dex3/right/state", HandState_); pub_rh.Init()
    low_msg = unitree_hg_msg_dds__LowState_()
    lh_msg, rh_msg = unitree_hg_msg_dds__HandState_(), unitree_hg_msg_dds__HandState_()

    def publish_lowstate():
        q, dq = d.qpos[body_q], d.qvel[body_v]
        tau_est = d.actuator_force[:n_body]
        for i in range(n_body):
            ms = low_msg.motor_state[i]
            ms.q, ms.dq, ms.tau_est, ms.mode = float(q[i]), float(dq[i]), float(tau_est[i]), 1
        low_msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        low_msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
        low_msg.mode_machine = args.mode_machine
        low_msg.mode_pr = 0
        low_msg.tick = int(d.time * 1000) & 0xFFFFFFFF
        low_msg.crc = crc.Crc(low_msg)
        pub_low.Write(low_msg)

    hold_q = np.zeros(n_body)
    viewer = None
    if args.viewer:
        import mujoco.viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    print(f"[mock] G1 mock robot running: dt={m.opt.timestep * 1000:.1f} ms, lowstate {args.state_hz:.0f} Hz, domain={args.domain}, "
          f"iface={args.iface or 'default'}, mode_machine={args.mode_machine}")
    print("[mock] waiting for rt/lowcmd (debug mode) or rt/arm_sdk (--motion) and rt/dex3/*/cmd ...")

    dt = m.opt.timestep
    state_every = max(1, int(round(1.0 / (dt * args.state_hz))))
    hand_every = max(1, int(round(1.0 / (dt * 100.0))))
    step = 0
    t_start = time.perf_counter()
    t_next = t_start
    t_report, sim_report = t_start, d.time
    last_view = 0.0
    max_err, phys_ms, view_ms = 0.0, [], []
    prev_counts = {k: 0 for k in ("low", "sdk", "lh", "rh")}
    mode_used = "hold"
    try:
        while viewer is None or viewer.is_running():
            now = time.perf_counter()
            t0 = now
            # ---------- motor commands for this step ----------
            q, dq = d.qpos[body_q], d.qvel[body_v]
            kp, kd = HOLD_KP.copy(), HOLD_KD.copy()
            q_cmd, dq_cmd, tau_cmd = hold_q.copy(), np.zeros(n_body), np.zeros(n_body)
            if lowcmd.fresh(now):
                mode_used = "lowcmd"
                with lowcmd.lock:
                    on = lowcmd.mode[:n_body] == 1
                    kp = np.where(on, lowcmd.kp[:n_body], 0.0)
                    kd = np.where(on, lowcmd.kd[:n_body], 1.0)  # disabled motor -> damping
                    q_cmd = np.where(on, lowcmd.q[:n_body], q)
                    dq_cmd = np.where(on, lowcmd.dq[:n_body], 0.0)
                    tau_cmd = np.where(on, lowcmd.tau[:n_body], 0.0)
            elif armsdk.fresh(now):
                mode_used = "arm_sdk"
                with armsdk.lock:
                    w = float(np.clip(armsdk.q[ARM_SDK_WEIGHT_ID], 0.0, 1.0))
                    ids = WAIST_IDS + ARM_IDS
                    kp[ids] = w * armsdk.kp[ids] + (1 - w) * HOLD_KP[ids]
                    kd[ids] = w * armsdk.kd[ids] + (1 - w) * HOLD_KD[ids]
                    q_cmd[ids] = w * armsdk.q[ids] + (1 - w) * hold_q[ids]
                    dq_cmd[ids] = w * armsdk.dq[ids]
                    tau_cmd[ids] = w * armsdk.tau[ids]
            else:
                mode_used = "hold"
            hands = []
            for store, qa in ((lhcmd, lh_q), (rhcmd, rh_q)):
                if store.fresh(now):
                    with store.lock:
                        hands.append((store.kp.copy(), store.kd.copy(), store.q.copy(), store.dq.copy(), store.tau.copy()))
                else:
                    hands.append((np.full(n_hand, HAND_HOLD_KP), np.full(n_hand, HAND_HOLD_KD),
                                  np.zeros(n_hand), np.zeros(n_hand), np.zeros(n_hand)))
            all_kp = np.concatenate([kp, hands[0][0], hands[1][0]])
            all_kd = np.concatenate([kd, hands[0][1], hands[1][1]])
            all_q = np.concatenate([q_cmd, hands[0][2], hands[1][2]])
            all_dq = np.concatenate([dq_cmd, hands[0][3], hands[1][3]])
            all_tau = np.concatenate([tau_cmd, hands[0][4], hands[1][4]])
            # force = kp*ctrl - kp*q - kd*qdot + (tau + kd*dq_cmd)
            m.actuator_gainprm[:, 0] = all_kp
            m.actuator_biasprm[:, 0] = all_tau + all_kd * all_dq
            m.actuator_biasprm[:, 1] = -all_kp
            m.actuator_biasprm[:, 2] = -all_kd
            d.ctrl[:] = all_q

            mujoco.mj_step(m, d)  # the passive viewer renders from its own copy made in sync()
            step += 1
            if mode_used != "hold":
                max_err = max(max_err, float(np.max(np.abs(q_cmd[ARM_IDS] - d.qpos[body_q][ARM_IDS]))))

            # ---------- publish states ----------
            if step % state_every == 0:
                publish_lowstate()
            if step % hand_every == 0:
                for msg, qa, va, pub in ((lh_msg, lh_q, lh_v, pub_lh), (rh_msg, rh_q, rh_v, pub_rh)):
                    for i in range(n_hand):
                        msg.motor_state[i].q = float(d.qpos[qa[i]])
                        msg.motor_state[i].dq = float(d.qvel[va[i]])
                    pub.Write(msg)
            phys_ms.append((time.perf_counter() - t0) * 1000)

            if viewer is not None and now - last_view > 1.0 / args.viewer_fps:
                tv0 = time.perf_counter()
                viewer.sync()
                view_ms.append((time.perf_counter() - tv0) * 1000)
                last_view = now

            # ---------- report ----------
            if now - t_report >= 2.0:
                el = now - t_report
                rtf = (d.time - sim_report) / el
                counts = {"low": lowcmd.count, "sdk": armsdk.count, "lh": lhcmd.count, "rh": rhcmd.count}
                hz = {k: (counts[k] - prev_counts[k]) / el for k in counts}
                prev_counts = counts
                line = (f"[mock] {mode_used:7s} | lowcmd {hz['low']:5.0f} Hz  arm_sdk {hz['sdk']:5.0f} Hz"
                        f"  dex3 L/R {hz['lh']:4.0f}/{hz['rh']:4.0f} Hz | step {np.mean(phys_ms):4.2f} ms"
                        f"  realtime x{rtf:4.2f}")
                if view_ms:
                    line += f" | viewer {len(view_ms) / el:4.1f} fps, sync {np.mean(view_ms):5.1f} ms"
                if mode_used == "lowcmd":
                    line += f" | kp arm/wrist/leg {lowcmd.kp[15]:.0f}/{lowcmd.kp[19]:.0f}/{lowcmd.kp[0]:.0f}"
                elif mode_used == "arm_sdk":
                    line += f" | weight {armsdk.q[ARM_SDK_WEIGHT_ID]:.2f}"
                if mode_used != "hold":
                    line += f" | max arm err {np.degrees(max_err):5.1f} deg"
                errs = (f"crc_err {lowcmd.crc_err + armsdk.crc_err}  mode_machine_mismatch "
                        f"{lowcmd.mm_mismatch + armsdk.mm_mismatch}  nan {lowcmd.nan + armsdk.nan + lhcmd.nan + rhcmd.nan}")
                print(line + " | " + errs)
                t_report, sim_report, max_err, phys_ms, view_ms = now, d.time, 0.0, [], []

            # ---------- real-time pacing ----------
            t_next += dt
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.1:
                t_next = time.perf_counter()  # fell behind: do not try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        print("[mock] stopped.")
        # skip interpreter teardown: DDS listener threads may still call into Python -> segfault on exit
        os._exit(0)


if __name__ == "__main__":
    main()
