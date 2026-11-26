"""
3D マルチIMU ESKF (四足・9 IMU + 足裏ToF)
=========================================
名目状態 (State):
  q      : 胴体姿勢クォータニオン (w,x,y,z), R_WB
  delta  : 関節たわみ (12,) = リンク角 − モータ角
  ddot   : たわみ速度 (12,)
  dddot  : たわみ角加速度 (12,)   ※ use_dddot=False なら常に0で誤差状態にも入らない
  bg     : ジャイロバイアス (9,3)  [胴体, FL大腿, FL下腿, FR..., RL..., RR...]
  p, v   : 胴体位置・速度(世界座標、z は接地面基準の高さ)
  ba     : 加速度計バイアス (9,3)

誤差状態 (use_dddot=True で 99 次元、False で 87 次元。添字は self.ix の各スライス):
  [dθ(3), dδ(12), dδ̇(12), (dδ̈(12)), db_g(27), dp(3), dv(3), db_a(27)]
  姿勢誤差は右乗算: q_true = q_nom ⊗ Exp(dθ)

予測 (毎サンプル):
  胴体ジャイロで姿勢、胴体加速度計で p, v (INS)。
  たわみは定加速度モデル δ += δ̇dt + ½δ̈dt², δ̇ += δ̈dt、δ̈ はジャークのランダムウォーク
  (use_dddot=False なら δ̇ のランダムウォーク)。

更新 (update_every サンプルに1回、51次元をまとめて1回):
 (A) リンクジャイロ (8 IMU × 3): ω_T = R2ᵀ(R1ᵀω_B + a1 q̇1) + a2 q̇2,  ω_S = R3ᵀω_T + a3 q̇3
 (B) リンク加速度計 (8 IMU × 3): 胴体加速度計から胴体点加速度を復元し、角速度・角加速度
     (q̈ = q̈m + δ̈)で運動学チェーンを伝播して各IMUの比力を予測。
     この観測は胴体姿勢に依存しない(R_WB と重力が打ち消し合う)。
 (C) 胴体傾斜 (3): 胴体加速度計の重力方向。緩い更新。
 ロバスト化: IMU・センサごとの3次元ブロックの正規化イノベーションが χ²(3) の閾値を
 超えたら、そのブロックの R を膨らませる(着地の衝撃など)。飽和した加速度計は使わない。

ToF更新 (tof_every サンプルに1回): 平面 z=0 までの距離。Huber型に R を膨らませる。

ヤコビアンは全て数値微分(boxplus摂動)。角加速度・モータ角加速度はゼロ位相平滑化した
数値微分を外部から供給(オンラインでは固定ラグFIR微分器で代替)。
"""

from typing import NamedTuple

import numpy as np
from quad_geom import (G, G_VEC, HIP_OFFSETS, JOINT_AXES, L1, L2, D1, D2,
                       DOWN, leg_rotations,
                       TOF_MOUNTS, TOF_RAY, TOF_MAX_TILT, TOF_BUDGET)

NL = 4           # 脚数
NJ = 12          # 関節数
NIMU = 9
CHI2_3_99 = 11.34   # χ²(3) の 99% 点


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


def cross(a, b):
    """3次元外積。np.crossは1回あたりのオーバーヘッドが大きく、
    _h_legで実行時間の約8割を占めていたため手書きにした。"""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


# ---------------------------------------------------------------- state
class State(NamedTuple):
    """名目状態。check_model.py などで真値を入れるときは必要な項目だけ渡せばよい。"""
    q: np.ndarray = np.array([1.0, 0, 0, 0])
    delta: np.ndarray = np.zeros(NJ)
    ddot: np.ndarray = np.zeros(NJ)
    dddot: np.ndarray = np.zeros(NJ)
    bg: np.ndarray = np.zeros((NIMU, 3))
    p: np.ndarray = np.zeros(3)
    v: np.ndarray = np.zeros(3)
    ba: np.ndarray = np.zeros((NIMU, 3))


def _error_layout(use_dddot):
    sizes = [('th', 3), ('d', NJ), ('dd', NJ)] + ([('ddd', NJ)] if use_dddot else []) \
        + [('bg', 3 * NIMU), ('p', 3), ('v', 3), ('ba', 3 * NIMU)]
    ix, k = {}, 0
    for name, n in sizes:
        ix[name] = slice(k, k + n)
        k += n
    return ix, k


# ---------------------------------------------------------------- estimator
class MultiImuESKF3D:
    def __init__(self, dt,
                 gyro_noise=0.02,
                 link_gyro_sigma=0.03,
                 link_acc_sigma=0.3,
                 incl_sigma=0.5, incl_dyn_weight=5.0,
                 ddot_accel_sigma=40.0,
                 use_dddot=True, dddot_jerk_sigma=3e4, ddot_accel_sigma_with_dddot=5.0,
                 bias_walk=2e-4,
                 accel_noise=0.08, tof_sigma=0.003,
                 acc_bias_walk=5e-4,
                 gyro_bias_sigma0=0.03, acc_bias_sigma0=0.15,
                 robust=True, acc_sat=None):
        """use_dddot      : たわみ角加速度 δ̈ を状態に入れる(物理的なSEA振動・着地を扱う)
        dddot_jerk_sigma : δ̈ のランダムウォーク強さ [rad/s³]
        robust           : リンクIMU観測のブロックごとのロバスト化
        acc_sat          : 加速度計の飽和値 [m/s²]。|成分| がこれ以上のIMUの加速度は使わない
        gyro_bias_sigma0, acc_bias_sigma0 : バイアスの初期σ。静置校正済みなら小さくする"""
        self.dt = dt
        self.use_dddot = use_dddot
        self.robust = robust
        self.acc_sat = acc_sat
        self.ix, self.nerr = _error_layout(use_dddot)
        ix, n = self.ix, self.nerr
        self.x = State(q=np.array([1.0, 0, 0, 0]))
        # 共分散
        P = np.zeros((n, n))
        def blk(M, name, val):
            s = ix[name]
            M[s, s] = np.eye(s.stop - s.start) * val
        blk(P, 'th', 0.2 ** 2)
        blk(P, 'd', 0.05 ** 2)
        blk(P, 'dd', 0.5 ** 2)
        if use_dddot:
            blk(P, 'ddd', 20.0 ** 2)
        blk(P, 'bg', gyro_bias_sigma0 ** 2)
        blk(P, 'p', 0.2 ** 2)
        blk(P, 'v', 0.3 ** 2)
        blk(P, 'ba', acc_bias_sigma0 ** 2)
        self.P = P
        # プロセスノイズ
        Q = np.zeros((n, n))
        blk(Q, 'th', (gyro_noise * dt) ** 2)
        blk(Q, 'd', 1e-10)
        if use_dddot:
            blk(Q, 'dd', (ddot_accel_sigma_with_dddot * dt) ** 2)
            blk(Q, 'ddd', (dddot_jerk_sigma * dt) ** 2)
        else:
            blk(Q, 'dd', (ddot_accel_sigma * dt) ** 2)
        blk(Q, 'bg', (bias_walk ** 2) * dt)
        blk(Q, 'p', 1e-12)
        blk(Q, 'v', (accel_noise * dt) ** 2)          # INS速度伝播
        blk(Q, 'ba', (acc_bias_walk ** 2) * dt)
        self.Q = Q
        self.R_gyro = link_gyro_sigma ** 2
        self.R_acc = link_acc_sigma ** 2
        self.incl_R0 = incl_sigma ** 2
        self.incl_dyn = incl_dyn_weight
        self.R_tof = tof_sigma ** 2
        self._leg_idx = [self._leg_err_idx(leg) for leg in range(NL)]

    # 互換用のプロパティ(run_estimator3d・デモが参照する)
    q = property(lambda s: s.x.q)
    delta = property(lambda s: s.x.delta)
    bg = property(lambda s: s.x.bg)
    p = property(lambda s: s.x.p)

    # ---------------------------------------------------------- boxplus
    def _boxplus(self, err, x=None):
        x = self.x if x is None else x
        ix = self.ix
        return State(
            q=quat_mul(x.q, quat_from_rotvec(err[ix['th']])),
            delta=x.delta + err[ix['d']],
            ddot=x.ddot + err[ix['dd']],
            dddot=x.dddot + err[ix['ddd']] if self.use_dddot else x.dddot,
            bg=x.bg + err[ix['bg']].reshape(NIMU, 3),
            p=x.p + err[ix['p']],
            v=x.v + err[ix['v']],
            ba=x.ba + err[ix['ba']].reshape(NIMU, 3))

    def _inject(self, err):
        x = self._boxplus(err)
        self.x = x._replace(q=x.q / np.linalg.norm(x.q))

    def _perturbed(self, i, eps):
        e = np.zeros(self.nerr)
        e[i] = eps
        return self._boxplus(e)

    # ---------------------------------------------------------- predict
    def predict(self, gyro0, f0):
        dt, x, ix = self.dt, self.x, self.ix
        w = gyro0 - x.bg[0]
        R_WB = quat_to_R(x.q)
        f0c = f0 - x.ba[0]
        a_W = R_WB @ f0c + G_VEC                     # INS力学化
        q = quat_mul(x.q, quat_from_rotvec(w * dt))
        self.x = x._replace(
            q=q / np.linalg.norm(q),
            delta=x.delta + x.ddot * dt + 0.5 * x.dddot * dt ** 2,
            ddot=x.ddot + x.dddot * dt,
            p=x.p + x.v * dt + 0.5 * a_W * dt ** 2,
            v=x.v + a_W * dt)

        F = np.eye(self.nerr)
        th, bg0 = ix['th'], slice(ix['bg'].start, ix['bg'].start + 3)
        ba0 = slice(ix['ba'].start, ix['ba'].start + 3)
        F[th, th] = quat_to_R(quat_from_rotvec(w * dt)).T
        F[th, bg0] = -np.eye(3) * dt                 # 胴体ジャイロバイアス
        F[ix['d'], ix['dd']] = np.eye(NJ) * dt       # δ <- δ̇
        if self.use_dddot:
            F[ix['d'], ix['ddd']] = np.eye(NJ) * 0.5 * dt ** 2   # δ <- δ̈
            F[ix['dd'], ix['ddd']] = np.eye(NJ) * dt             # δ̇ <- δ̈
        F[ix['p'], ix['v']] = np.eye(3) * dt         # p <- v
        F[ix['v'], th] = -R_WB @ skew(f0c) * dt      # v <- 姿勢誤差
        F[ix['v'], ba0] = -R_WB * dt                 # v <- 胴体加速度計バイアス
        self.P = F @ self.P @ F.T + self.Q

    # ---------------------------------------------------------- 観測モデル
    def _h_leg(self, x, leg, f0, alpha_B, qm, qm_dot, qm_ddot):
        """1脚分の [gyro_T(3), gyro_S(3), acc_T(3), acc_S(3)] を予測 (12,)。
        x は State。胴体角速度は状態ではなく self._w_B_input(生の胴体ジャイロ)を使う。"""
        R_WB = quat_to_R(x.q)
        w_B = self._w_B_input - x.bg[0]
        wW = R_WB @ w_B
        aW = R_WB @ alpha_B
        a_B = R_WB @ (f0 - x.ba[0]) + G_VEC

        j = 3 * leg
        qj = qm[j:j+3] + x.delta[j:j+3]
        qdj = qm_dot[j:j+3] + x.ddot[j:j+3]
        qddj = qm_ddot[j:j+3] + x.dddot[j:j+3]
        R1, R12, R123 = leg_rotations(qj)
        a1, a2, a3 = JOINT_AXES

        R2 = R1.T @ R12
        R3 = R12.T @ R123
        w_T = R2.T @ (R1.T @ w_B + a1 * qdj[0]) + a2 * qdj[1]
        w_S = R3.T @ w_T + a3 * qdj[2]
        g_T = w_T + x.bg[1 + 2 * leg]
        g_S = w_S + x.bg[2 + 2 * leg]

        u1 = R_WB @ a1
        u2 = R_WB @ (R1 @ a2)
        wT_W = wW + qdj[0] * u1 + qdj[1] * u2
        aT_W = (aW + qddj[0] * u1 + qddj[1] * u2
                + cross(wW, qdj[0] * u1)
                + cross(wW + qdj[0] * u1, qdj[1] * u2))
        R_WT = R_WB @ R12
        u3 = R_WT @ a3
        wS_W = wT_W + qdj[2] * u3
        aS_W = aT_W + qddj[2] * u3 + cross(wT_W, qdj[2] * u3)
        R_WS = R_WB @ R123

        r_hip = R_WB @ HIP_OFFSETS[leg]
        a_hip = a_B + cross(aW, r_hip) + cross(wW, cross(wW, r_hip))
        r_t = R_WT @ (D1 * DOWN)
        a_it = a_hip + cross(aT_W, r_t) + cross(wT_W, cross(wT_W, r_t))
        r_k = R_WT @ (L1 * DOWN)
        a_kn = a_hip + cross(aT_W, r_k) + cross(wT_W, cross(wT_W, r_k))
        r_s = R_WS @ (D2 * DOWN)
        a_is = a_kn + cross(aS_W, r_s) + cross(wS_W, cross(wS_W, r_s))

        f_T = R_WT.T @ (a_it - G_VEC) + x.ba[1 + 2 * leg]
        f_S = R_WS.T @ (a_is - G_VEC) + x.ba[2 + 2 * leg]
        return np.concatenate([g_T, g_S, f_T, f_S])

    def _leg_err_idx(self, leg):
        """脚legの観測が依存する誤差状態インデックス。
        胴体姿勢 dθ は含めない: _h_leg の全項は R_WB·(胴体座標の量) の形で、
        R_WB と重力が最後の R_WTᵀ(a − g) で打ち消し合うため、∂h_leg/∂dθ ≡ 0。"""
        ix, j = self.ix, 3 * leg
        idx = []
        for name in (['d', 'dd', 'ddd'] if self.use_dddot else ['d', 'dd']):
            idx += list(range(ix[name].start + j, ix[name].start + j + 3))
        for name in ('bg', 'ba'):
            s = ix[name].start
            for imu in (0, 1 + 2 * leg, 2 + 2 * leg):
                idx += list(range(s + 3 * imu, s + 3 * imu + 3))
        return np.array(idx)

    def update(self, gyro, acc, qm, qm_dot, alpha_B, qm_ddot):
        """gyro:(9,3), acc:(9,3)。リンクIMU観測 + 胴体傾斜の統合更新。
        ヤコビアンは脚ごとのブロック疎構造を利用した数値微分。"""
        if self.acc_sat is not None and np.any(np.abs(acc[0]) >= self.acc_sat):
            return   # 胴体加速度計が飽和していると全予測が壊れる
        self._w_B_input = gyro[0]
        f0 = acc[0]
        eps = 1e-6
        n = self.nerr

        z = np.zeros(48)
        h0 = np.zeros(48)
        H = np.zeros((48, n))
        Rd = np.zeros(48)
        args = (f0, alpha_B, qm, qm_dot, qm_ddot)
        for leg in range(NL):
            k = 12 * leg
            z[k:k+3] = gyro[1 + 2 * leg]
            z[k+3:k+6] = gyro[2 + 2 * leg]
            z[k+6:k+9] = acc[1 + 2 * leg]
            z[k+9:k+12] = acc[2 + 2 * leg]
            h_leg0 = self._h_leg(self.x, leg, *args)
            h0[k:k+12] = h_leg0
            for i in self._leg_idx[leg]:
                H[k:k+12, i] = (self._h_leg(self._perturbed(i, eps), leg, *args)
                                - h_leg0) / eps
            Rd[k:k+6] = self.R_gyro
            Rd[k+6:k+12] = self.R_acc
            if self.acc_sat is not None:
                for kk, imu in ((6, 1 + 2 * leg), (9, 2 + 2 * leg)):
                    if np.any(np.abs(acc[imu]) >= self.acc_sat):
                        Rd[k+kk:k+kk+3] = 1e6      # 飽和: 実質使わない

        # --- 胴体傾斜(重力方向) ---
        th = self.ix['th']
        ba0 = range(self.ix['ba'].start, self.ix['ba'].start + 3)
        R_WB = quat_to_R(self.x.q)
        h_inc = R_WB.T @ (-G_VEC) + self.x.ba[0]
        H_inc = np.zeros((3, n))
        for i in list(range(th.start, th.stop)) + list(ba0):
            xp = self._perturbed(i, eps)
            H_inc[:, i] = (quat_to_R(xp.q).T @ (-G_VEC) + xp.ba[0] - h_inc) / eps
        r_inc = (self.incl_R0
                 + self.incl_dyn * ((np.linalg.norm(f0) - G) / G) ** 2) * G ** 2

        y = np.concatenate([z - h0, f0 - h_inc])
        H_all = np.vstack([H, H_inc])
        r_all = np.concatenate([Rd, np.full(3, r_inc)])
        PHt = self.P @ H_all.T
        HPHt = H_all @ PHt
        if self.robust:
            # 3次元ブロック(IMU×センサ)ごとの正規化イノベーションで R を膨らませる
            for b in range(16):
                s = slice(3 * b, 3 * b + 3)
                Sb = HPHt[s, s] + np.diag(r_all[s])
                nis = y[s] @ np.linalg.solve(Sb, y[s])
                if nis > CHI2_3_99:
                    r_all[s] *= nis / CHI2_3_99
        S = HPHt + np.diag(r_all)
        K = np.linalg.solve(S, PHt.T).T
        self._inject(K @ y)
        I_KH = np.eye(n) - K @ H_all
        self.P = I_KH @ self.P @ I_KH.T + (K * r_all) @ K.T

    # ---------------------------------------------------------- ToF更新
    def _h_tof(self, x, leg, qm):
        """脚legの足裏ToF測距値を予測。無効条件なら None。
        接地面は z=0 の平面と仮定(不整地は地面高さ推定の拡張が必要)。"""
        R_WB = quat_to_R(x.q)
        j = 3 * leg
        qj = qm[j:j+3] + x.delta[j:j+3]
        _, R12, R123 = leg_rotations(qj)
        R_WS = R_WB @ R123
        p_tof = (x.p + R_WB @ HIP_OFFSETS[leg]
                 + (R_WB @ R12) @ (L1 * DOWN) + R_WS @ TOF_MOUNTS[leg])
        u = R_WS @ TOF_RAY
        if u[2] > -TOF_MAX_TILT:      # 足裏が地面をほぼ向いていない
            return None
        return -p_tof[2] / u[2]

    def update_tof(self, tof, qm):
        """tof: (4,) 各足のToF測距値 [m]。NaN=無効(範囲外など)。"""
        eps = 1e-6
        ix = self.ix
        rows_h, rows_z, rows_H = [], [], []
        for leg in range(NL):
            if not np.isfinite(tof[leg]):
                continue
            h = self._h_tof(self.x, leg, qm)
            if h is None or h < -0.1 or h > 1.0:
                continue
            # 依存する誤差状態: 姿勢, この脚のδ, 位置
            j = 3 * leg
            idx = (list(range(ix['th'].start, ix['th'].stop))
                   + list(range(ix['d'].start + j, ix['d'].start + j + 3))
                   + list(range(ix['p'].start, ix['p'].stop)))
            Hrow = np.zeros(self.nerr)
            for i in idx:
                hp = self._h_tof(self._perturbed(i, eps), leg, qm)
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
        y = z - h
        # 外れ値の扱い: 3σを超えたイノベーションは捨てずに R を膨らませて弱く使う
        # (Huber型)。捨てるゲートだと、推定σが小さいときに遅れ由来の大きな
        # イノベーションが続くと全て棄却され、推定がずれたまま戻れなくなる。
        r = np.full(len(h), self.R_tof)
        s_diag = np.einsum('ij,jk,ik->i', H, self.P, H) + r
        nis = y ** 2 / s_diag
        r = r * np.maximum(1.0, nis / 9.0)
        R = np.diag(r)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(len(y)))
        self._inject(K @ y)
        I_KH = np.eye(self.nerr) - K @ H
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
                    use_tof=True, tof_every=10, tof_lag=TOF_BUDGET / 2, **kw):
    """tof_lag: ToF測定値の代表時刻の遅れ [s]。測定値はタイミングバジェット区間の
    平均なので、区間中央(budget/2前)の関節角で予測値を計算する
    (速く動くのは脚なので、関節角だけ遡れば大部分の遅れが取れる)。
    kw は MultiImuESKF3D に渡す。"""
    dt = data['dt']
    lag = int(round(tof_lag / dt))
    n = len(data['t'])
    ekf = MultiImuESKF3D(dt=dt, **kw)

    # 角加速度(胴体, 胴体座標)とモータ角加速度を前処理
    alpha_B = np.gradient(smooth_zero_phase(data['gyro'][:, 0], dt, smooth_hz), dt, axis=0)
    qm_ddot = np.gradient(smooth_zero_phase(data['qm_dot'], dt, smooth_hz), dt, axis=0)

    has_tof = use_tof and ('tof' in data)
    if has_tof:
        # 初期高さz: 最初の有効ToFから逆算(ガウス・ニュートン法3回)
        i0 = np.argmax(np.isfinite(data['tof']).any(axis=1))
        tof0, qm0 = data['tof'][i0], data['qm'][max(0, i0 - lag)]
        for _ in range(3):
            num = den = 0.0
            for leg in range(NL):
                if not np.isfinite(tof0[leg]):
                    continue
                h = ekf._h_tof(ekf.x, leg, qm0)
                if h is None:
                    continue
                pz = ekf.x.p.copy()
                pz[2] += 1e-4
                hp = ekf._h_tof(ekf.x._replace(p=pz), leg, qm0)
                dh = (hp - h) / 1e-4
                num += dh * (tof0[leg] - h)
                den += dh * dh
            if den > 0:
                pz = ekf.x.p.copy()
                pz[2] += num / den
                ekf.x = ekf.x._replace(p=pz)
    quat_hist = np.zeros((n, 4))
    delta_hist = np.zeros((n, NJ))
    ddot_hist = np.zeros((n, NJ))
    dddot_hist = np.zeros((n, NJ))
    bias_hist = np.zeros((n, NIMU, 3))
    pos_hist = np.zeros((n, 3))
    for i in range(n):
        ekf.predict(data['gyro'][i, 0], data['acc'][i, 0])
        if i % update_every == 0:
            ekf.update(data['gyro'][i], data['acc'][i],
                       data['qm'][i], data['qm_dot'][i],
                       alpha_B[i], qm_ddot[i])
        if has_tof and i % tof_every == 0:
            ekf.update_tof(data['tof'][i], data['qm'][max(0, i - lag)])
        x = ekf.x
        quat_hist[i] = x.q
        delta_hist[i] = x.delta
        ddot_hist[i] = x.ddot
        dddot_hist[i] = x.dddot
        bias_hist[i] = x.bg
        pos_hist[i] = x.p
    return {'quat': quat_hist, 'delta': delta_hist, 'ddot': ddot_hist,
            'dddot': dddot_hist, 'bias': bias_hist, 'pos': pos_hist}
