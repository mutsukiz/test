"""
PyBulletバックエンド(動力学シミュレーション)
==============================================
データ形式は sim3d_numpy.py と互換(demo3d.py --pybullet / check_model.py --pybullet)。
PyBulletは Python 3.10〜3.12 なら配布バイナリで入る(3.13はソースビルドが失敗した)。

構成
- 直列弾性アクチュエータ(SEA): モータ位置 qm は指令 q_cmd に2次系(ωm, 臨界減衰)で
  追従する。ばねトルク τ = k(qm − q) + c(q̇m − q̇) をリンク側の関節に掛ける。
  エンコーダはモータ側 qm を読む。たわみ真値 δ = q − qm。
- 制御器は controller(t, obs) -> q_cmd(12,) の形の呼び出し可能オブジェクトで差し替えられる。
    TrotController     : 足先軌道トロット + 逆運動学 + 姿勢フィードバック
                         + Raibert着地位置補正 + SEAたわみのフィードフォワード補償
    SineJointController: numpy版と同じ関節正弦波(歩行にならない。比較用)
  強化学習の方策を使う場合は、obs(下記)から q_cmd を返す関数を渡せばよい。
- 補助(Assist): 水平の弱いばね・減衰、yawの減衰、傾きが guard_angle を超えたときだけ
  働く姿勢ガード、任意の鉛直持ち上げ。掛かった力・トルクは data['assist_*'] に残る。
- 足裏ToF: rayTestBatch で実地形にレイキャスト(ロボット自身は衝突フィルタで除外)。
  測定値は quad_geom.tof_measure(VL53L4CDモデル: タイミングバジェット平均+ノイズ)。
- IMU: 物理1 kHzで位置を2階差分し、ゼロ位相3タップ平均 [1/4, 1/2, 1/4] をかけてから
  500 Hzに間引く(センサ内部のデシメーションフィルタ相当)。

obs(制御器に渡す辞書、すべて真値)
  t, rpy(3), quat_wxyz(4), omega_B(3, 胴体座標), v_B(3, 胴体座標), pos(3),
  q(12, リンク側), qd(12), qm(12, モータ側), qm_dot(12), contact(4, bool)
"""

import numpy as np
import pybullet as p
import pybullet_data

from quad_geom import (G, G_VEC, HIP_OFFSETS, L1, L2, D1, D2, DOWN, LEG_NAMES,
                       FOOT_R, leg_rotations, TOF_MOUNTS, TOF_RAY, TOF_MAX_TILT,
                       TOF_RANGE, TOF_SIGMA, TOF_BUDGET, TOF_OFFSET_MAG,
                       tof_measure)

NL, NJ, NIMU = 4, 12, 9
ROBOT_GROUP = 4      # ロボットの衝突グループ。ToFレイはこれを無視する
RAY_MASK = 1 | 2     # レイが当たる対象(地面などロボット以外)
TROT_OFFSET = np.array([0.0, 0.5, 0.5, 0.0])   # FL, FR, RL, RR の歩容位相


# =================================================================== kinematics
def leg_fk(q3):
    """関節角(3,) -> 股関節から見た足先の球の中心(胴体座標)"""
    _, R12, R123 = leg_rotations(q3)
    return R12 @ (L1 * DOWN) + R123 @ (L2 * DOWN)


def leg_ik(pf):
    """股関節から見た足先の球の中心(胴体座標) -> 関節角(3,)。膝は後ろ折れ(q3<0)。
    外転: Rx(q1)[x', 0, -r] = [x', r sin q1, -r cos q1] より q1 = atan2(y, -z)。
    脚平面内: X = -x, Y = r として X = L1 s2 + L2 s23, Y = L1 c2 + L2 c23 の2リンクIK。"""
    x, y, z = pf
    q1 = np.arctan2(y, -z)
    r = np.hypot(y, z)
    c3 = np.clip((x * x + r * r - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
    q3 = -np.arccos(c3)
    q2 = np.arctan2(-x, r) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
    return np.array([q1, q2, q3])


def leg_jacobian(q3, eps=1e-6):
    J = np.zeros((3, 3))
    f0 = leg_fk(q3)
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        J[:, i] = (leg_fk(q3 + dq) - f0) / eps
    return J


# =================================================================== controllers
class SineJointController:
    """sim3d_numpy.py と同一の関節正弦波(歩行にはならない。比較用)"""

    def __init__(self, gait_hz=1.5):
        self.f = gait_hz

    def __call__(self, t, obs):
        f = self.f
        q = np.zeros(NJ)
        for leg, ph in enumerate([0.0, np.pi, np.pi, 0.0]):
            j = 3 * leg
            q[j + 0] = 0.05 * np.sin(2 * np.pi * f * t + ph + 0.5)
            q[j + 1] = 0.50 + 0.25 * np.sin(2 * np.pi * f * t + ph)
            q[j + 2] = -1.00 + 0.35 * np.sin(2 * np.pi * f * t + ph + 0.8)
        return q


class TrotController:
    """足先軌道によるトロット。

    足先目標(股関節基準、胴体座標):
      支持期(位相 s∈[0,1)): x = stride/2·(1 − 2s), z = −(h − FOOT_R) + Δz_att
      遊脚期(位相 s∈[0,1)): x は −stride/2 から着地点まで余弦補間、z に step_height·sin(πs)
      着地点 = stride/2 + k_raibert·(v_x − vx)、横方向 y も k_raibert·v_y
    姿勢フィードバック(支持脚の長さ調整。小角度で足先の世界高さ ≈ z + roll·y − pitch·x):
      Δz_i = k_att·(roll·y_i − pitch·x_i) + d_att·(ω_x·y_i − ω_y·x_i)
    SEAたわみのフィードフォワード: 支持脚に F = Rᵀ[0,0,m g/n_stance] が掛かるとして
      静的つり合い k(qm − q) + Jᵀ F = 0 より q_cmd = q_des − Jᵀ F / k。
    """

    def __init__(self, gait_hz=2.0, duty=0.5, body_height=0.33, step_height=0.05,
                 vx=0.0, k_att=1.5, d_att=0.08, k_raibert=0.05, ff_gain=1.0,
                 k_joint=(80.0, 60.0, 40.0), mass=9.0, start_time=0.0,
                 ramp_time=1.0):
        self.f = gait_hz
        self.duty = duty
        self.h = body_height
        self.step_h = step_height
        self.vx = vx
        self.k_att, self.d_att = k_att, d_att
        self.k_r = k_raibert
        self.ff = ff_gain
        self.k = np.asarray(k_joint, float)
        self.mass = mass
        self.t0 = start_time
        self.ramp = ramp_time

    def nominal_q(self):
        return np.concatenate([leg_ik(np.array([0.0, 0.0, -(self.h - FOOT_R)]))
                               for _ in range(NL)])

    def __call__(self, t, obs):
        tt = t - self.t0
        amp = np.clip(tt / self.ramp, 0.0, 1.0) if self.ramp > 0 else 1.0
        roll, pitch, _ = obs['rpy']
        wx, wy, _ = obs['omega_B']
        vB = obs['v_B']
        T_st = self.duty / self.f
        stride = self.vx * T_st * amp
        z0 = -(self.h - FOOT_R)

        stance = np.zeros(NL, bool)
        targets = []
        for leg in range(NL):
            xh, yh = HIP_OFFSETS[leg, :2]
            ph = (self.f * tt + TROT_OFFSET[leg]) % 1.0 if tt > 0 else 0.0
            dz = (self.k_att * (roll * yh - pitch * xh)
                  + self.d_att * (wx * yh - wy * xh))
            if tt <= 0 or ph < self.duty:
                s = ph / self.duty if tt > 0 else 0.5
                x = stride / 2 * (1 - 2 * s)
                y = 0.0
                z = z0 + dz
                stance[leg] = True
            else:
                s = (ph - self.duty) / (1 - self.duty)
                x_land = stride / 2 + amp * self.k_r * (vB[0] - self.vx)
                y_land = amp * self.k_r * vB[1]
                b = (1 - np.cos(np.pi * s)) / 2
                x = -stride / 2 + (x_land + stride / 2) * b
                y = y_land * b
                z = z0 + amp * self.step_h * np.sin(np.pi * s)
            targets.append(np.array([x, y, z]))

        q_cmd = np.zeros(NJ)
        n_st = max(1, stance.sum())
        R = _rpy_to_R(obs['rpy'])
        F = R.T @ np.array([0.0, 0.0, self.mass * G / n_st])
        for leg in range(NL):
            q = leg_ik(targets[leg])
            if stance[leg] and self.ff > 0:
                q = q - self.ff * (leg_jacobian(q).T @ F) / self.k
            q_cmd[3 * leg:3 * leg + 3] = q
        return q_cmd


def _rpy_to_R(rpy):
    return np.array(p.getMatrixFromQuaternion(
        p.getQuaternionFromEuler(list(rpy)))).reshape(3, 3)


# =================================================================== assist
class Assist:
    """試験リグ相当の補助。既定値は「歩行の邪魔をしない程度」に弱くしてある。
    lift_frac   : 鉛直上向きの一定力(体重比)。既定0
    k_xy, d_xy  : 原点への水平ばね[N/m]・減衰[Ns/m](その場トロットで流されない程度)
    d_yaw       : yaw角速度の減衰[Nms/rad]
    guard_angle : roll/pitch がこれを超えた分だけ k_guard, d_guard で戻す[rad]
    """

    def __init__(self, lift_frac=0.0, k_xy=30.0, d_xy=20.0, d_yaw=0.3,
                 guard_angle=0.15, k_guard=80.0, d_guard=5.0, mass=9.0,
                 enabled=True):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def wrench(self, pos, vel, rpy, omega_W):
        if not self.enabled:
            return np.zeros(3), np.zeros(3)
        F = np.array([-self.k_xy * pos[0] - self.d_xy * vel[0],
                      -self.k_xy * pos[1] - self.d_xy * vel[1],
                      self.lift_frac * self.mass * G])
        T = np.zeros(3)
        T[2] = -self.d_yaw * omega_W[2]
        for i in range(2):
            ex = np.sign(rpy[i]) * max(0.0, abs(rpy[i]) - self.guard_angle)
            if ex != 0.0:
                T[i] = -self.k_guard * ex - self.d_guard * omega_W[i]
        return F, T


# =================================================================== URDF
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
            f'<geometry><sphere radius="{FOOT_R}"/></geometry></collision>'
            f'<visual><origin xyz="0 0 {-L2/2}"/>'
            f'<geometry><cylinder radius="0.015" length="{L2}"/></geometry></visual>'
            '</link>')
    parts.append('</robot>')
    with open(path, 'w') as fp:
        fp.write(''.join(parts))
    return path


ROBOT_MASS = 5.0 + NL * (0.10 + 0.6 + 0.3)


# =================================================================== simulator
class QuadSimulatorPyBullet:
    def __init__(self, duration=10.0, sim_hz=1000, out_hz=500, ctrl_hz=500,
                 controller=None, assist=None, gui=False, seed=0,
                 accel_bias_mag=0.0, settle_time=1.5):
        self.duration = duration
        self.sim_dt = 1.0 / sim_hz
        self.sub = sim_hz // out_hz
        self.dt = 1.0 / out_hz
        self.ctrl_every = max(1, sim_hz // ctrl_hz)
        self.gui = gui
        self.settle_time = settle_time
        self.rng = np.random.default_rng(seed)

        # SEA / モータ
        self.k_joint = np.array([80.0, 60.0, 40.0])   # [外転, 股P, 膝] Nm/rad
        self.c_joint = np.array([1.5, 1.2, 0.8])      # Nms/rad
        self.motor_hz = 25.0                          # モータ位置追従の帯域
        self.tau_max = 25.0                           # トルク上限 [Nm]
        self.foot_contact_stiffness = 3e4             # 足裏の接触剛性 [N/m]
        self.foot_contact_damping = 3e2               # 足裏の接触減衰 [Ns/m]

        self.controller = controller if controller is not None else TrotController(
            k_joint=self.k_joint, mass=ROBOT_MASS)
        self.assist = assist if assist is not None else Assist(mass=ROBOT_MASS)

        # センサ(numpy版と同一)
        self.gyro_noise = 0.02
        self.acc_noise = 0.05
        self.enc_quant = 0.001
        self.enc_rate_noise = 0.01
        self.gyro_bias_true = self.rng.uniform(-0.03, 0.03, size=(NIMU, 3))
        self.acc_bias_true = self.rng.uniform(-1, 1, size=(NIMU, 3)) * accel_bias_mag
        self.acc_range = 16 * G                       # 加速度計レンジ ±16 g で飽和
        self.gyro_range = np.deg2rad(2000.0)          # ジャイロレンジ ±2000 dps
        self.tof_sigma = TOF_SIGMA
        self.tof_budget = TOF_BUDGET
        self.tof_offset_mag = TOF_OFFSET_MAG

    # ------------------------------------------------------------------
    def _setup(self):
        cid = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -G)
        p.setTimeStep(self.sim_dt)
        plane = p.loadURDF('plane.urdf')
        h0 = getattr(self.controller, 'h', 0.33)
        robot = p.loadURDF(build_urdf(), basePosition=[0, 0, h0 + 0.005],
                           useFixedBase=False)
        names = {p.getJointInfo(robot, ji)[1].decode(): ji
                 for ji in range(p.getNumJoints(robot))}
        self.jidx = [names[f'{nm}_{j}'] for nm in LEG_NAMES
                     for j in ('abd', 'hip', 'knee')]
        self.link_thigh = [names[f'{nm}_hip'] for nm in LEG_NAMES]
        self.link_shank = [names[f'{nm}_knee'] for nm in LEG_NAMES]
        for li in range(-1, p.getNumJoints(robot)):
            # ToFレイが自分の足(レイ始点の近く)に当たらないようにする
            p.setCollisionFilterGroupMask(robot, li, ROBOT_GROUP, -1)
        for li in self.link_shank:
            # 足裏ゴム相当の柔らかい接触(剛体接触だと着地で数百 m/s² の衝撃が出る)
            p.changeDynamics(robot, li, lateralFriction=0.9,
                             contactStiffness=self.foot_contact_stiffness,
                             contactDamping=self.foot_contact_damping)
        p.changeDynamics(plane, -1, lateralFriction=0.9)
        for ji in self.jidx:
            p.setJointMotorControl2(robot, ji, p.VELOCITY_CONTROL, force=0)
        return cid, plane, robot

    def _obs(self, robot, plane, t):
        bp, bo = p.getBasePositionAndOrientation(robot)
        bv, bw = p.getBaseVelocity(robot)
        R = np.array(p.getMatrixFromQuaternion(bo)).reshape(3, 3)
        js = p.getJointStates(robot, self.jidx)
        contact = np.array([len(p.getContactPoints(robot, plane, li, -1)) > 0
                            for li in self.link_shank])
        return {'t': t, 'pos': np.array(bp), 'quat_wxyz': np.array([bo[3], *bo[:3]]),
                'rpy': np.array(p.getEulerFromQuaternion(bo)),
                'omega_B': R.T @ np.array(bw), 'v_B': R.T @ np.array(bv),
                'omega_W': np.array(bw), 'v_W': np.array(bv), 'R': R,
                'q': np.array([s[0] for s in js]), 'qd': np.array([s[1] for s in js]),
                'qm': self.qm.copy(), 'qm_dot': self.qm_dot.copy(),
                'contact': contact}

    def _step(self, robot, plane, t, k_ctrl):
        obs = self._obs(robot, plane, t)
        if k_ctrl % self.ctrl_every == 0:
            self.q_cmd = np.asarray(self.controller(t, obs), float)
        # モータ: 臨界減衰2次系で q_cmd に追従
        wm = 2 * np.pi * self.motor_hz
        qmdd = wm ** 2 * (self.q_cmd - self.qm) - 2 * wm * self.qm_dot
        self.qm_dot = self.qm_dot + qmdd * self.sim_dt
        self.qm = self.qm + self.qm_dot * self.sim_dt
        # SEAばねトルク
        k = np.tile(self.k_joint, NL)
        c = np.tile(self.c_joint, NL)
        tau = k * (self.qm - obs['q']) + c * (self.qm_dot - obs['qd'])
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        p.setJointMotorControlArray(robot, self.jidx, p.TORQUE_CONTROL, forces=tau)
        F, T = self.assist.wrench(obs['pos'], obs['v_W'], obs['rpy'], obs['omega_W'])
        if F.any():
            p.applyExternalForce(robot, -1, F.tolist(), obs['pos'].tolist(), p.WORLD_FRAME)
        if T.any():
            p.applyExternalTorque(robot, -1, T.tolist(), p.WORLD_FRAME)
        p.stepSimulation()
        return obs, F, T

    # ------------------------------------------------------------------
    def run(self):
        cid, plane, robot = self._setup()
        q0 = (self.controller.nominal_q() if hasattr(self.controller, 'nominal_q')
              else np.asarray(self.controller(0.0, None), float))
        for ji, qq in zip(self.jidx, q0):
            p.resetJointState(robot, ji, qq)
        self.qm = q0.copy()
        self.qm_dot = np.zeros(NJ)
        self.q_cmd = q0.copy()

        # --- 整定: 制御器を t<0 の時刻で回す(TrotControllerは4脚支持で立つ) ---
        n_settle = int(self.settle_time / self.sim_dt)
        for i in range(n_settle):
            self._step(robot, plane, (i - n_settle) * self.sim_dt, i)

        # --- 本計測(物理1 kHzで記録) ---
        n_sim = int(self.duration / self.sim_dt)
        pos = np.zeros((n_sim, NIMU, 3))
        Rm = np.zeros((n_sim, NIMU, 3, 3))
        omg = np.zeros((n_sim, NIMU, 3))
        quat = np.zeros((n_sim, 4))
        q_link = np.zeros((n_sim, NJ))
        qm = np.zeros((n_sim, NJ))
        qm_dot = np.zeros((n_sim, NJ))
        q_cmd = np.zeros((n_sim, NJ))
        contact = np.zeros((n_sim, NL), bool)
        d_tof = np.full((n_sim, NL), np.nan)
        assist_F = np.zeros((n_sim, 3))
        assist_T = np.zeros((n_sim, 3))
        links = self.link_thigh + self.link_shank
        for i in range(n_sim):
            t = i * self.sim_dt
            obs, F, T = self._step(robot, plane, t, n_settle + i)
            assist_F[i], assist_T[i] = F, T
            q_cmd[i] = self.q_cmd
            # --- ステップ後の状態を記録 ---
            bp, bo = p.getBasePositionAndOrientation(robot)
            _, bw = p.getBaseVelocity(robot)
            pos[i, 0] = bp
            Rm[i, 0] = np.array(p.getMatrixFromQuaternion(bo)).reshape(3, 3)
            omg[i, 0] = bw
            quat[i] = [bo[3], bo[0], bo[1], bo[2]]
            js = p.getJointStates(robot, self.jidx)
            q_link[i] = [s[0] for s in js]
            qm[i], qm_dot[i] = self.qm, self.qm_dot
            lss = p.getLinkStates(robot, links, computeLinkVelocity=1,
                                  computeForwardKinematics=1)
            ray_from, ray_to = [], []
            for leg in range(NL):
                for kk, dd in ((0, D1), (1, D2)):
                    ls = lss[leg + NL * kk]
                    pf = np.array(ls[4])          # URDFリンク座標原点(=関節位置)
                    Rf = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
                    k_imu = 1 + 2 * leg + kk
                    pos[i, k_imu] = pf + Rf @ (dd * DOWN)
                    Rm[i, k_imu] = Rf
                    omg[i, k_imu] = ls[7]
                    if kk == 1:
                        p_mount = pf + Rf @ TOF_MOUNTS[leg]
                        u = Rf @ TOF_RAY
                        ray_from.append(p_mount)
                        ray_to.append(p_mount + u * TOF_RANGE[1] * 1.5
                                      if u[2] < -TOF_MAX_TILT else p_mount)
            if i % self.sub == 0:
                contact[i] = [len(p.getContactPoints(robot, plane, li, -1)) > 0
                              for li in self.link_shank]
                hits = p.rayTestBatch(ray_from, ray_to, collisionFilterMask=RAY_MASK)
                for leg in range(NL):
                    fr = hits[leg][2]
                    if 0.0 < fr < 1.0:
                        d_tof[i, leg] = fr * np.linalg.norm(
                            np.subtract(ray_to[leg], ray_from[leg]))
        p.disconnect(cid)

        # --- IMU合成: 1 kHzで微分 -> ゼロ位相3タップ平均 -> 500 Hzに間引き ---
        vel = np.gradient(pos, self.sim_dt, axis=0)
        accel = np.gradient(vel, self.sim_dt, axis=0)
        f_sens = np.einsum('nkji,nkj->nki', Rm, accel - G_VEC)
        w_sens = np.einsum('nkji,nkj->nki', Rm, omg)

        def lp3(x):
            y = x.copy()
            y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
            return y
        idx = np.arange(0, n_sim, self.sub)
        n = len(idx)
        gyro = lp3(w_sens)[idx]
        acc = lp3(f_sens)[idx]
        gyro += self.gyro_bias_true[None]
        gyro += self.gyro_noise * self.rng.standard_normal((n, NIMU, 3))
        acc += self.acc_bias_true[None]
        acc += self.acc_noise * self.rng.standard_normal((n, NIMU, 3))
        self.acc_saturated = np.abs(acc) >= self.acc_range
        acc = np.clip(acc, -self.acc_range, self.acc_range)
        gyro = np.clip(gyro, -self.gyro_range, self.gyro_range)

        qm_s = qm[idx]
        qm_meas = np.round(qm_s / self.enc_quant) * self.enc_quant
        qm_dot_meas = qm_dot[idx] + self.enc_rate_noise * self.rng.standard_normal((n, NJ))
        tof = tof_measure(d_tof[idx], self.dt, self.rng, sigma=self.tof_sigma,
                          budget=self.tof_budget, offset_mag=self.tof_offset_mag)

        return {
            't': idx * self.sim_dt, 'dt': self.dt,
            'gyro': gyro, 'acc': acc,
            'qm': qm_meas, 'qm_dot': qm_dot_meas,
            'quat_true': quat[idx], 'delta_true': q_link[idx] - qm_s,
            'gyro_bias_true': self.gyro_bias_true,
            'tof': tof, 'pos_true': pos[idx, 0],
            # 解析用
            'tof_true': d_tof[idx], 'contact': contact[idx], 'q_cmd': q_cmd[idx],
            'assist_force': assist_F[idx], 'assist_torque': assist_T[idx],
        }


if __name__ == '__main__':
    data = QuadSimulatorPyBullet(duration=5.0).run()
    print({k: getattr(v, 'shape', v) for k, v in data.items() if hasattr(v, 'shape')})
    print('max |delta| [deg]:', np.rad2deg(np.abs(data['delta_true']).max()))
