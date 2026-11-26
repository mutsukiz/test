"""
PyBulletバックエンド(動力学シミュレーション)
==============================================
※ このモジュールは開発環境にpybulletが無いため未実行です。
   データ形式は sim3d_numpy.py と完全互換で、demo3d.py --pybullet で使用。
   問題があれば「確認ポイント」のコメントを参照してください。

構成:
- 直列弾性アクチュエータ(SEA)モデル: 各関節をトルク制御し
  τ = k(q_cmd - q) + c(q̇_cmd - q̇) を印加。q_cmd がモータ角(エンコーダ)、
  PyBulletの関節角 q がリンク角。たわみ真値 δ = q - q_cmd。
- 胴体は自由浮遊。歩行制御器なしで直立を保つため、柔らかい仮想ハーネス
  (ばね-ダンパの外力/トルク)を胴体に印加する(試験リグ相当。
  harness=Falseで無効化可能だが転倒しない制御が別途必要)。
- 地面との接触はPyBulletが解決。足先は下腿末端の球。

確認ポイント(実機環境で最初に見るべき点):
1. getLinkStateのリンク座標系が想定(大腿/下腿の-z方向にリンクが伸びる)
   と一致しているか -> demo3d.pyの整合性チェックで残差が小さいこと
2. ハーネス剛性が強すぎると脚負荷が減りたわみが小さくなる
3. sim周波数(1kHz)とSEA剛性の数値安定性(発振したらc増 or dt減)
"""

import numpy as np
import pybullet as p
import pybullet_data

from quad_geom import (G, G_VEC, HIP_OFFSETS, L1, L2, D1, D2, DOWN, LEG_NAMES,
                       TOF_MOUNT, TOF_RAY, TOF_SIGMA, TOF_RANGE, TOF_MAX_TILT)

NL, NJ, NIMU = 4, 12, 9


def trot_trajectory(t, f=1.5):
    """sim3d_numpy.pyと同一のトロット様モータ軌道 (12,) と速度 (12,)"""
    leg_phase = np.array([0.0, np.pi, np.pi, 0.0])
    qm = np.zeros((len(t), NJ))
    for leg in range(NL):
        ph = leg_phase[leg]
        j = 3 * leg
        qm[:, j+0] = 0.05 * np.sin(2*np.pi*f*t + ph + 0.5)
        qm[:, j+1] = 0.50 + 0.25 * np.sin(2*np.pi*f*t + ph)
        qm[:, j+2] = -1.00 + 0.35 * np.sin(2*np.pi*f*t + ph + 0.8)
    qm_dot = np.gradient(qm, t[1] - t[0], axis=0)
    return qm, qm_dot


def build_urdf(path='quad_sea.urdf'):
    """4脚12関節のURDFを生成(リンクは各リンク座標の-z方向に伸びる)"""
    def inertial(m, ixx, iyy, izz, oz=0.0):
        return (f'<inertial><origin xyz="0 0 {oz}"/><mass value="{m}"/>'
                f'<inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" '
                f'iyz="0" izz="{izz}"/></inertial>')

    parts = ['<?xml version="1.0"?><robot name="quad_sea">']
    parts.append(
        '<link name="base">' + inertial(5.0, 0.02, 0.06, 0.07) +
        '<collision><geometry><box size="0.40 0.20 0.10"/></geometry></collision>'
        '<visual><geometry><box size="0.40 0.20 0.10"/></geometry></visual>'
        '</link>')
    for leg in range(NL):
        nm = LEG_NAMES[leg]
        ox, oy, oz = HIP_OFFSETS[leg]
        parts.append(
            f'<joint name="{nm}_abd" type="revolute">'
            f'<parent link="base"/><child link="{nm}_hipmount"/>'
            f'<origin xyz="{ox} {oy} {oz}"/><axis xyz="1 0 0"/>'
            f'<limit lower="-1.0" upper="1.0" effort="60" velocity="30"/></joint>'
            f'<link name="{nm}_hipmount">' + inertial(0.10, 2e-4, 2e-4, 2e-4) +
            '</link>'
            f'<joint name="{nm}_hip" type="revolute">'
            f'<parent link="{nm}_hipmount"/><child link="{nm}_thigh"/>'
            f'<origin xyz="0 0 0"/><axis xyz="0 1 0"/>'
            f'<limit lower="-2.5" upper="2.5" effort="60" velocity="30"/></joint>'
            f'<link name="{nm}_thigh">' + inertial(0.6, 3e-3, 3e-3, 3e-4, oz=-L1/2) +
            f'<collision><origin xyz="0 0 {-L1/2}"/>'
            f'<geometry><cylinder radius="0.02" length="{L1}"/></geometry></collision>'
            f'<visual><origin xyz="0 0 {-L1/2}"/>'
            f'<geometry><cylinder radius="0.02" length="{L1}"/></geometry></visual>'
            '</link>'
            f'<joint name="{nm}_knee" type="revolute">'
            f'<parent link="{nm}_thigh"/><child link="{nm}_shank"/>'
            f'<origin xyz="0 0 {-L1}"/><axis xyz="0 1 0"/>'
            f'<limit lower="-2.7" upper="0.5" effort="60" velocity="30"/></joint>'
            f'<link name="{nm}_shank">' + inertial(0.3, 1.5e-3, 1.5e-3, 1e-4, oz=-L2/2) +
            f'<collision><origin xyz="0 0 {-L2}"/>'
            f'<geometry><sphere radius="0.02"/></geometry></collision>'
            f'<visual><origin xyz="0 0 {-L2/2}"/>'
            f'<geometry><cylinder radius="0.015" length="{L2}"/></geometry></visual>'
            '</link>')
    parts.append('</robot>')
    with open(path, 'w') as fp:
        fp.write(''.join(parts))
    return path


class QuadSimulatorPyBullet:
    def __init__(self, duration=10.0, sim_hz=1000, out_hz=500,
                 gui=False, harness=True, seed=0, accel_bias_mag=0.0):
        self.duration = duration
        self.sim_dt = 1.0 / sim_hz
        self.sub = sim_hz // out_hz
        self.dt = 1.0 / out_hz
        self.gui = gui
        self.harness = harness
        self.rng = np.random.default_rng(seed)

        # SEA パラメータ(numpy版のk_jointと対応)
        self.k_joint = np.array([80.0, 60.0, 40.0])
        self.c_joint = np.array([1.5, 1.2, 0.8])

        # センサ(numpy版と同一)
        self.gyro_noise = 0.02
        self.acc_noise = 0.05
        self.enc_quant = 0.001
        self.enc_rate_noise = 0.01
        self.gyro_bias_true = self.rng.uniform(-0.03, 0.03, size=(NIMU, 3))
        self.acc_bias_true = self.rng.uniform(
            -1, 1, size=(NIMU, 3)) * accel_bias_mag

    # ------------------------------------------------------------------
    def run(self):
        cid = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -G)
        p.setTimeStep(self.sim_dt)
        p.loadURDF('plane.urdf')
        robot = p.loadURDF(build_urdf(), basePosition=[0, 0, 0.36],
                           useFixedBase=False)

        # 関節名 -> index、リンクindex(childリンク=jointと同index)
        joint_ids = {}
        for ji in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, ji)
            joint_ids[info[1].decode()] = ji
        jidx = []      # (12,) 関節index
        link_thigh, link_shank = [], []
        for nm in LEG_NAMES:
            jidx += [joint_ids[f'{nm}_abd'], joint_ids[f'{nm}_hip'],
                     joint_ids[f'{nm}_knee']]
            link_thigh.append(joint_ids[f'{nm}_hip'])
            link_shank.append(joint_ids[f'{nm}_knee'])

        # デフォルトの速度モータを無効化(トルク制御にする)
        for ji in jidx:
            p.setJointMotorControl2(robot, ji, p.VELOCITY_CONTROL, force=0)

        # --- 初期整定(1s): 初期姿勢を指令し、着地させる ---
        t_settle = np.arange(0, 1.0, self.sim_dt)
        qm0, _ = trot_trajectory(np.array([0.0]))
        for ji, q0 in zip(jidx, qm0[0]):
            p.resetJointState(robot, ji, q0)
        for _ in t_settle:
            self._sea_step(robot, jidx, qm0[0], np.zeros(NJ))
            self._apply_harness(robot)
            p.stepSimulation()

        # --- 本計測 ---
        n_sim = int(self.duration / self.sim_dt)
        t_sim = np.arange(n_sim) * self.sim_dt
        qm_cmd, qm_dot_cmd = trot_trajectory(t_sim)

        n = n_sim // self.sub
        t = np.arange(n) * self.dt
        pos = np.zeros((n, NIMU, 3))
        Rm = np.zeros((n, NIMU, 3, 3))
        omg = np.zeros((n, NIMU, 3))
        quat_true = np.zeros((n, 4))
        delta_true = np.zeros((n, NJ))
        tof = np.full((n, NL), np.nan)
        pos_true = np.zeros((n, 3))

        rec = 0
        for i in range(n_sim):
            self._sea_step(robot, jidx, qm_cmd[i], qm_dot_cmd[i])
            self._apply_harness(robot)
            p.stepSimulation()

            if i % self.sub == 0:
                bp, bo = p.getBasePositionAndOrientation(robot)
                bv, bw = p.getBaseVelocity(robot)
                Rb = np.array(p.getMatrixFromQuaternion(bo)).reshape(3, 3)
                pos[rec, 0] = bp
                Rm[rec, 0] = Rb
                omg[rec, 0] = bw
                quat_true[rec] = [bo[3], bo[0], bo[1], bo[2]]  # xyzw -> wxyz
                pos_true[rec] = bp

                js = p.getJointStates(robot, jidx)
                q_act = np.array([s[0] for s in js])
                delta_true[rec] = q_act - qm_cmd[i]

                ray_from, ray_to = [], []
                for leg in range(NL):
                    for kk, (li, dd) in enumerate([(link_thigh[leg], D1),
                                                   (link_shank[leg], D2)]):
                        ls = p.getLinkState(robot, li, computeLinkVelocity=1,
                                            computeForwardKinematics=1)
                        pf = np.array(ls[4])   # URDFリンク座標系原点(=関節位置)
                        Rf = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
                        wf = np.array(ls[7])   # 世界座標角速度
                        k_imu = 1 + 2 * leg + kk
                        pos[rec, k_imu] = pf + Rf @ (dd * DOWN)
                        Rm[rec, k_imu] = Rf
                        omg[rec, k_imu] = wf
                        if kk == 1:  # 下腿: 足裏ToFのレイを準備(実地形を反映)
                            # pf = 下腿リンク原点 = 膝関節位置。
                            # TOF_MOUNT(=L2*DOWN)は膝からの足先オフセット。
                            p_mount = pf + Rf @ TOF_MOUNT
                            u = Rf @ TOF_RAY
                            if u[2] < -TOF_MAX_TILT:
                                ray_from.append(p_mount)
                                ray_to.append(p_mount + u * TOF_RANGE[1] * 1.5)
                            else:
                                ray_from.append(p_mount)
                                ray_to.append(p_mount)  # ゼロ長 = 無効

                # --- 足裏ToF: 実地形へのレイキャスト(pybullet.rayTestBatch) ---
                hits = p.rayTestBatch(ray_from, ray_to)
                for leg in range(NL):
                    hit_frac = hits[leg][2]
                    if 0.0 < hit_frac < 1.0:
                        u = np.array(ray_to[leg]) - np.array(ray_from[leg])
                        d = hit_frac * np.linalg.norm(u)
                        d += TOF_SIGMA * self.rng.standard_normal()
                        if TOF_RANGE[0] < d < TOF_RANGE[1]:
                            tof[rec, leg] = d
                rec += 1
        p.disconnect(cid)

        # --- 後処理: IMU出力合成(numpy版と同一手順) ---
        vel = np.gradient(pos, self.dt, axis=0)
        accel = np.gradient(vel, self.dt, axis=0)
        gyro = np.zeros((n, NIMU, 3))
        acc = np.zeros((n, NIMU, 3))
        for i in range(n):
            for k in range(NIMU):
                gyro[i, k] = Rm[i, k].T @ omg[i, k]
                acc[i, k] = Rm[i, k].T @ (accel[i, k] - G_VEC)
        gyro += self.gyro_bias_true[None]
        gyro += self.gyro_noise * self.rng.standard_normal((n, NIMU, 3))
        acc += self.acc_bias_true[None]
        acc += self.acc_noise * self.rng.standard_normal((n, NIMU, 3))

        qm_sub = qm_cmd[::self.sub][:n]
        qm_dot_sub = qm_dot_cmd[::self.sub][:n]
        qm_meas = np.round(qm_sub / self.enc_quant) * self.enc_quant
        qm_dot_meas = (qm_dot_sub
                       + self.enc_rate_noise * self.rng.standard_normal((n, NJ)))

        return {
            't': t, 'dt': self.dt,
            'gyro': gyro, 'acc': acc,
            'qm': qm_meas, 'qm_dot': qm_dot_meas,
            'quat_true': quat_true, 'delta_true': delta_true,
            'gyro_bias_true': self.gyro_bias_true,
            'tof': tof, 'pos_true': pos_true,
        }

    # ------------------------------------------------------------------
    def _sea_step(self, robot, jidx, q_cmd, qd_cmd):
        js = p.getJointStates(robot, jidx)
        for m, ji in enumerate(jidx):
            k = self.k_joint[m % 3]
            c = self.c_joint[m % 3]
            q, qd = js[m][0], js[m][1]
            tau = k * (q_cmd[m] - q) + c * (qd_cmd[m] - qd)
            p.setJointMotorControl2(robot, ji, p.TORQUE_CONTROL, force=tau)

    def _apply_harness(self, robot):
        """柔らかい仮想ハーネス(試験リグ)。歩行制御器導入後は無効化する。"""
        if not self.harness:
            return
        bp, bo = p.getBasePositionAndOrientation(robot)
        bv, bw = p.getBaseVelocity(robot)
        F = 300.0 * (np.array([0, 0, 0.33]) - bp) - 60.0 * np.array(bv)
        F[2] = max(F[2], 0.0) * 0.3   # 鉛直は弱く(脚に荷重を残す)
        Rb = np.array(p.getMatrixFromQuaternion(bo)).reshape(3, 3)
        # 姿勢誤差(直立からの回転ベクトル近似)
        rot_err = 0.5 * np.array([Rb[2, 1] - Rb[1, 2],
                                  Rb[0, 2] - Rb[2, 0],
                                  Rb[1, 0] - Rb[0, 1]])
        T = -30.0 * rot_err - 6.0 * np.array(bw)
        p.applyExternalForce(robot, -1, F.tolist(), bp, p.WORLD_FRAME)
        p.applyExternalTorque(robot, -1, T.tolist(), p.WORLD_FRAME)


if __name__ == '__main__':
    data = QuadSimulatorPyBullet(duration=5.0, gui=False).run()
    print({k: getattr(v, 'shape', v) for k, v in data.items()
           if hasattr(v, 'shape')})
    print('max |delta| [deg]:', np.rad2deg(np.abs(data['delta_true']).max()))
