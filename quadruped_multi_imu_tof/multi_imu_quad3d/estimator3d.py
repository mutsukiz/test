"""
3D マルチIMU ESKF (四足・9 IMU)
================================
ノミナル状態:
  q_WB   : 胴体姿勢クォータニオン (w,x,y,z)
  delta  : 関節たわみ (12,)  = リンク角 - モータ角
  ddot   : たわみ速度 (12,)
  bg     : ジャイロバイアス (9,3)  [胴体 + 大腿x4 + 下腿x4]

誤差状態 (54,):
  [dθ(3), dδ(12), dδ̇(12), db(27)]
  姿勢誤差は右乗算: q_true = q_nom ⊗ Exp(dθ)

予測: 胴体ジャイロで姿勢積分、δ += δ̇ dt、δ̇/バイアスはランダムウォーク。

更新(100Hz, まとめて1回のEKF更新):
 (A) リンクジャイロ観測 (8x3): 運動学チェーンによる角速度伝播モデル
     ω_thigh = R2ᵀR1ᵀ(ω_B) + R2ᵀa1 q̇1 + a2 q̇2 + b_thigh
     ω_shank = R3ᵀ ω_thigh(バイアス除く) + a3 q̇3 + b_shank
     (q̇ = q̇m + δ̇。δ̇とバイアスの分離は(B)のδ絶対拘束を通じて可能)
 (B) リンク加速度計観測 (8x3): 胴体加速度計から胴体点加速度を復元し、
     角速度・角加速度で運動学チェーンを世界座標で伝播して各IMU位置の
     比力を予測(動的加速度補償)。残差は相対角=δに鋭敏。
 (C) 胴体傾斜(重力方向)の緩い更新: roll/pitchのドリフト抑制。
     yawは原理的に不可観測(磁気/視覚なしのため)で緩慢にドリフトする。

ヤコビアンは全て数値微分(boxplus摂動)で、導出ミスを排除している。
角加速度・モータ角加速度はゼロ位相平滑化した数値微分を外部から供給
(オンラインでは固定ラグFIR微分器で代替)。
"""

import numpy as np
from quad_geom import (G, G_VEC, HIP_OFFSETS, JOINT_AXES, L1, L2, D1, D2,
                       DOWN, leg_rotations,
                       TOF_MOUNT, TOF_RAY, TOF_RANGE, TOF_MAX_TILT)

NL = 4           # 脚数
NJ = 12          # 関節数
NIMU = 9
NERR = 3 + NJ + NJ + 3 * NIMU + 6 + 3 * NIMU   # 87 [dθ, dδ, dδ̇, db_g, dp, dv, db_a]


# ---------------------------------------------------------------- quaternion
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_from_rotvec(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([1.0, *(0.5 * v)])
    u = v / a
    return np.array([np.cos(a / 2), *(np.sin(a / 2) * u)])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------------------------------------------- estimator
class MultiImuESKF3D:
    def __init__(self, dt,
                 gyro_noise=0.02,
                 link_gyro_sigma=0.03,
                 link_acc_sigma=0.3,
                 incl_sigma=0.5, incl_dyn_weight=5.0,
                 ddot_accel_sigma=40.0,
                 bias_walk=2e-4,
                 accel_noise=0.08, tof_sigma=0.006,
                 acc_bias_walk=5e-4):
        self.dt = dt
        # ノミナル状態
        self.q = np.array([1.0, 0, 0, 0])
        self.delta = np.zeros(NJ)
        self.ddot = np.zeros(NJ)
        self.bg = np.zeros((NIMU, 3))
        self.p = np.zeros(3)      # 胴体位置(z=接地面基準の高さ)
        self.v = np.zeros(3)      # 胴体速度
        self.ba = np.zeros((NIMU, 3))   # 加速度計バイアス
        # 共分散
        self.P = np.zeros((NERR, NERR))
        self.P[0:3, 0:3] = np.eye(3) * 0.2 ** 2
        self.P[3:15, 3:15] = np.eye(NJ) * 0.05 ** 2
        self.P[15:27, 15:27] = np.eye(NJ) * 0.5 ** 2
        self.P[27:54, 27:54] = np.eye(27) * 0.03 ** 2
        self.P[54:57, 54:57] = np.eye(3) * 0.2 ** 2   # 位置
        self.P[57:60, 57:60] = np.eye(3) * 0.3 ** 2   # 速度
        self.P[60:87, 60:87] = np.eye(27) * 0.15 ** 2  # 加速度計バイアス
        # プロセスノイズ
        Q = np.zeros((NERR, NERR))
        Q[0:3, 0:3] = np.eye(3) * (gyro_noise * dt) ** 2
        Q[3:15, 3:15] = np.eye(NJ) * 1e-10
        Q[15:27, 15:27] = np.eye(NJ) * (ddot_accel_sigma * dt) ** 2
        Q[27:54, 27:54] = np.eye(27) * (bias_walk ** 2) * dt
        Q[54:57, 54:57] = np.eye(3) * 1e-12
        Q[57:60, 57:60] = np.eye(3) * (accel_noise * dt) ** 2  # INS速度伝播
        Q[60:87, 60:87] = np.eye(27) * (acc_bias_walk ** 2) * dt
        self.Q = Q
        self.R_gyro = link_gyro_sigma ** 2
        self.R_acc = link_acc_sigma ** 2
        self.incl_R0 = incl_sigma ** 2
        self.incl_dyn = incl_dyn_weight
        self.R_tof = tof_sigma ** 2

    # ---------------------------------------------------------- boxplus
    def _boxplus(self, err):
        q = quat_mul(self.q, quat_from_rotvec(err[0:3]))
        return (q,
                self.delta + err[3:15],
                self.ddot + err[15:27],
                self.bg + err[27:54].reshape(NIMU, 3),
                self.p + err[54:57],
                self.v + err[57:60],
                self.ba + err[60:87].reshape(NIMU, 3))

    def _inject(self, err):
        (self.q, self.delta, self.ddot, self.bg,
         self.p, self.v, self.ba) = self._boxplus(err)
        self.q /= np.linalg.norm(self.q)

    # ---------------------------------------------------------- predict
    def predict(self, gyro0, f0):
        dt = self.dt
        w = gyro0 - self.bg[0]
        R_WB = quat_to_R(self.q)
        f0c = f0 - self.ba[0]
        a_W = R_WB @ f0c + G_VEC                     # INS力学化
        self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)
        self.delta = self.delta + self.ddot * dt
        self.p = self.p + self.v * dt + 0.5 * a_W * dt ** 2
        self.v = self.v + a_W * dt

        F = np.eye(NERR)
        F[0:3, 0:3] = quat_to_R(quat_from_rotvec(w * dt)).T
        F[0:3, 27:30] = -np.eye(3) * dt              # 胴体ジャイロバイアス
        F[3:15, 15:27] = np.eye(NJ) * dt             # δ <- δ̇
        F[54:57, 57:60] = np.eye(3) * dt             # p <- v
        F[57:60, 0:3] = -R_WB @ skew(f0c) * dt       # v <- 姿勢誤差
        F[57:60, 60:63] = -R_WB * dt                 # v <- 胴体加速度計バイアス
        self.P = F @ self.P @ F.T + self.Q

    # ---------------------------------------------------------- 観測モデル
    def _h_leg(self, state, leg, f0, alpha_B, qm, qm_dot, qm_ddot):
        """1脚分の [gyro_T(3), gyro_S(3), acc_T(3), acc_S(3)] を予測 (12,)"""
        q, delta, ddot, bg, p, v, ba = state
        R_WB = quat_to_R(q)
        w_B = self._w_B_input - bg[0]
        wW = R_WB @ w_B
        aW = R_WB @ alpha_B
        a_B = R_WB @ (f0 - ba[0]) + G_VEC

        j = 3 * leg
        qj = qm[j:j+3] + delta[j:j+3]
        qdj = qm_dot[j:j+3] + ddot[j:j+3]
        qddj = qm_ddot[j:j+3]
        R1, R12, R123 = leg_rotations(qj)
        a1, a2, a3 = JOINT_AXES

        R2 = R1.T @ R12
        R3 = R12.T @ R123
        w_T = R2.T @ (R1.T @ w_B + a1 * qdj[0]) + a2 * qdj[1]
        w_S = R3.T @ w_T + a3 * qdj[2]
        g_T = w_T + bg[1 + 2 * leg]
        g_S = w_S + bg[2 + 2 * leg]

        u1 = R_WB @ a1
        u2 = R_WB @ (R1 @ a2)
        wT_W = wW + qdj[0] * u1 + qdj[1] * u2
        aT_W = (aW + qddj[0] * u1 + qddj[1] * u2
                + np.cross(wW, qdj[0] * u1)
                + np.cross(wW + qdj[0] * u1, qdj[1] * u2))
        R_WT = R_WB @ R12
        u3 = R_WT @ a3
        wS_W = wT_W + qdj[2] * u3
        aS_W = aT_W + qddj[2] * u3 + np.cross(wT_W, qdj[2] * u3)
        R_WS = R_WB @ R123

        r_hip = R_WB @ HIP_OFFSETS[leg]
        a_hip = a_B + np.cross(aW, r_hip) + np.cross(wW, np.cross(wW, r_hip))
        r_t = R_WT @ (D1 * DOWN)
        a_it = a_hip + np.cross(aT_W, r_t) + np.cross(wT_W, np.cross(wT_W, r_t))
        r_k = R_WT @ (L1 * DOWN)
        a_kn = a_hip + np.cross(aT_W, r_k) + np.cross(wT_W, np.cross(wT_W, r_k))
        r_s = R_WS @ (D2 * DOWN)
        a_is = a_kn + np.cross(aS_W, r_s) + np.cross(wS_W, np.cross(wS_W, r_s))

        f_T = R_WT.T @ (a_it - G_VEC) + ba[1 + 2 * leg]
        f_S = R_WS.T @ (a_is - G_VEC) + ba[2 + 2 * leg]
        return np.concatenate([g_T, g_S, f_T, f_S])

    def _leg_err_idx(self, leg):
        """脚legの観測が依存する誤差状態インデックス(18,)"""
        j = 3 * leg
        return np.array(
            list(range(0, 3))                       # 姿勢
            + list(range(3 + j, 3 + j + 3))         # δ
            + list(range(15 + j, 15 + j + 3))       # δ̇
            + list(range(27, 30))                   # 胴体バイアス
            + list(range(27 + 3 * (1 + 2 * leg), 30 + 3 * (1 + 2 * leg)))
            + list(range(27 + 3 * (2 + 2 * leg), 30 + 3 * (2 + 2 * leg)))
            + list(range(60, 63))                   # 胴体加速度計バイアス
            + list(range(60 + 3 * (1 + 2 * leg), 63 + 3 * (1 + 2 * leg)))
            + list(range(60 + 3 * (2 + 2 * leg), 63 + 3 * (2 + 2 * leg))))

    def update(self, gyro, acc, qm, qm_dot, alpha_B, qm_ddot):
        """gyro:(9,3), acc:(9,3)。リンクIMU観測 + 胴体傾斜の統合更新。
        ヤコビアンは脚ごとのブロック疎構造を利用した数値微分。"""
        self._w_B_input = gyro[0]
        f0 = acc[0]
        eps = 1e-6
        state0 = (self.q, self.delta, self.ddot, self.bg,
                  self.p, self.v, self.ba)

        z = np.zeros(48)
        h0 = np.zeros(48)
        H = np.zeros((48, NERR))
        args = (f0, alpha_B, qm, qm_dot, qm_ddot)
        for leg in range(NL):
            k = 12 * leg
            z[k:k+3] = gyro[1 + 2 * leg]
            z[k+3:k+6] = gyro[2 + 2 * leg]
            z[k+6:k+9] = acc[1 + 2 * leg]
            z[k+9:k+12] = acc[2 + 2 * leg]
            h_leg0 = self._h_leg(state0, leg, *args)
            h0[k:k+12] = h_leg0
            for i in self._leg_err_idx(leg):
                e = np.zeros(NERR)
                e[i] = eps
                H[k:k+12, i] = (self._h_leg(self._boxplus(e), leg, *args)
                                - h_leg0) / eps

        Rd = np.zeros(48)
        for leg in range(NL):
            k = 12 * leg
            Rd[k:k+6] = self.R_gyro
            Rd[k+6:k+12] = self.R_acc

        # --- 胴体傾斜(重力方向) ---
        R_WB = quat_to_R(self.q)
        h_inc = R_WB.T @ (-G_VEC) + self.ba[0]
        H_inc = np.zeros((3, NERR))
        for i in [0, 1, 2, 60, 61, 62]:
            e = np.zeros(NERR)
            e[i] = eps
            qb, _, _, _, _, _, bab = self._boxplus(e)
            H_inc[:, i] = (quat_to_R(qb).T @ (-G_VEC) + bab[0] - h_inc) / eps
        r_inc = (self.incl_R0
                 + self.incl_dyn * ((np.linalg.norm(f0) - G) / G) ** 2) * G ** 2

        z_all = np.concatenate([z, f0])
        h_all = np.concatenate([h0, h_inc])
        H_all = np.vstack([H, H_inc])
        R_all = np.diag(np.concatenate([Rd, np.full(3, r_inc)]))

        y = z_all - h_all
        S = H_all @ self.P @ H_all.T + R_all
        K = self.P @ H_all.T @ np.linalg.solve(S, np.eye(len(y)))
        self._inject(K @ y)
        I_KH = np.eye(NERR) - K @ H_all
        self.P = I_KH @ self.P @ I_KH.T + K @ R_all @ K.T

    # ---------------------------------------------------------- ToF更新
    def _h_tof(self, state, leg, qm):
        """脚legの足裏ToF測距値を予測。無効条件なら None。
        接地面は z=0 の平面と仮定(不整地は地面高さ推定の拡張が必要)。"""
        q, delta, ddot, bg, p, v, ba = state
        R_WB = quat_to_R(q)
        j = 3 * leg
        qj = qm[j:j+3] + delta[j:j+3]
        _, R12, R123 = leg_rotations(qj)
        R_WS = R_WB @ R123
        p_foot = (p + R_WB @ HIP_OFFSETS[leg]
                  + (R_WB @ R12) @ (L1 * DOWN) + R_WS @ TOF_MOUNT)
        u = R_WS @ TOF_RAY
        if u[2] > -TOF_MAX_TILT:      # 足裏が地面をほぼ向いていない
            return None
        return -p_foot[2] / u[2]

    def update_tof(self, tof, qm):
        """tof: (4,) 各足のToF測距値 [m]。NaN=無効(接地中/範囲外)。"""
        state0 = (self.q, self.delta, self.ddot, self.bg,
                  self.p, self.v, self.ba)
        eps = 1e-6
        rows_h, rows_z, rows_H = [], [], []
        for leg in range(NL):
            if not np.isfinite(tof[leg]):
                continue
            h = self._h_tof(state0, leg, qm)
            if h is None or h < -0.1 or h > 1.0:
                continue
            # 依存する誤差状態: 姿勢, この脚のδ, 位置
            j = 3 * leg
            idx = [0, 1, 2, 3 + j, 4 + j, 5 + j, 54, 55, 56]
            Hrow = np.zeros(NERR)
            for i in idx:
                e = np.zeros(NERR)
                e[i] = eps
                hp = self._h_tof(self._boxplus(e), leg, qm)
                if hp is None:
                    continue
                Hrow[i] = (hp - h) / eps
            rows_h.append(h)
            rows_z.append(tof[leg])
            rows_H.append(Hrow)
        if not rows_h:
            return
        h = np.array(rows_h)
        z = np.array(rows_z)
        H = np.array(rows_H)
        R = np.eye(len(h)) * self.R_tof
        y = z - h
        # 外れ値ゲート(3σ, 地面の凹凸や誤検出対策)
        S = H @ self.P @ H.T + R
        keep = np.abs(y) < 3.0 * np.sqrt(np.diag(S))
        if not keep.any():
            return
        y, H, S = y[keep], H[keep], S[np.ix_(keep, keep)]
        R = np.eye(keep.sum()) * self.R_tof
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(len(y)))
        self._inject(K @ y)
        I_KH = np.eye(NERR) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

# -------------------------------------------------------------------- utils
def smooth_zero_phase(x, dt, hz=30.0):
    """前進+後退の指数平滑(ゼロ位相)。オンラインは固定ラグFIRで代替可。"""
    a = dt / (dt + 1.0 / (2 * np.pi * hz))
    y = x.astype(float).copy()
    for i in range(1, len(y)):
        y[i] = y[i-1] + a * (y[i] - y[i-1])
    for i in range(len(y) - 2, -1, -1):
        y[i] = y[i+1] + a * (y[i] - y[i+1])
    return y


def run_estimator3d(data, update_every=3, smooth_hz=30.0,
                    use_tof=True, tof_every=10, **kw):
    dt = data['dt']
    n = len(data['t'])
    ekf = MultiImuESKF3D(dt=dt, **kw)

    # 角加速度(胴体, 胴体座標)とモータ角加速度を前処理
    alpha_B = np.gradient(smooth_zero_phase(data['gyro'][:, 0], dt, smooth_hz), dt, axis=0)
    qm_ddot = np.gradient(smooth_zero_phase(data['qm_dot'], dt, smooth_hz), dt, axis=0)

    has_tof = use_tof and ('tof' in data)
    if has_tof:
        # 初期高さz: 最初の有効ToFから逆算(ニュートン法3回)
        i0 = np.argmax(np.isfinite(data['tof']).any(axis=1))
        tof0, qm0 = data['tof'][i0], data['qm'][i0]
        for _ in range(3):
            state = (ekf.q, ekf.delta, ekf.ddot, ekf.bg,
                     ekf.p, ekf.v, ekf.ba)
            num = den = 0.0
            for leg in range(NL):
                if not np.isfinite(tof0[leg]):
                    continue
                h = ekf._h_tof(state, leg, qm0)
                if h is None:
                    continue
                pz = ekf.p.copy(); pz[2] += 1e-4
                hp = ekf._h_tof((ekf.q, ekf.delta, ekf.ddot, ekf.bg,
                                 pz, ekf.v, ekf.ba), leg, qm0)
                dh = (hp - h) / 1e-4
                num += dh * (tof0[leg] - h)
                den += dh * dh
            if den > 0:
                ekf.p[2] += num / den
    quat_hist = np.zeros((n, 4))
    delta_hist = np.zeros((n, NJ))
    bias_hist = np.zeros((n, NIMU, 3))
    pos_hist = np.zeros((n, 3))
    for i in range(n):
        ekf.predict(data['gyro'][i, 0], data['acc'][i, 0])
        if i % update_every == 0:
            ekf.update(data['gyro'][i], data['acc'][i],
                       data['qm'][i], data['qm_dot'][i],
                       alpha_B[i], qm_ddot[i])
        if has_tof and i % tof_every == 0:
            ekf.update_tof(data['tof'][i], data['qm'][i])
        quat_hist[i] = ekf.q
        delta_hist[i] = ekf.delta
        bias_hist[i] = ekf.bg
        pos_hist[i] = ekf.p
    return {'quat': quat_hist, 'delta': delta_hist, 'bias': bias_hist,
            'pos': pos_hist}
