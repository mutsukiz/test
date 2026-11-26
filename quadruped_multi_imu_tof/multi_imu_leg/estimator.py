"""
マルチIMU + エンコーダ融合 Error-State EKF
==========================================
状態量 x = [theta0, delta1, delta2, b0, b1, b2]  (6次元)
  theta0 : 胴体ピッチ角
  delta_i: 関節iのたわみ(リンク角 - モータ角) <- 低剛性ズレの直接推定
  b_j    : IMU jのジャイロバイアス

予測(propagation): ジャイロとエンコーダ速度を入力に積分
  theta0'  = w0 - b0
  delta1'  = (w1 - b1) - (w0 - b0) - qm1_dot
  delta2'  = (w2 - b2) - (w1 - b1) - qm2_dot
  (隣接リンクのジャイロ差分 - モータ速度 = たわみ速度)

更新(update): 2種類
 (1) 胴体傾斜更新: 胴体加速度計の重力方向。動的加速度で常に汚れるが
     周期運動では平均的に不偏なので、大きな観測分散で緩く当てる
     (theta0とb0のドリフト抑制用)。
 (2) リンク比力残差更新(本手法の核):
     胴体加速度計から胴体点の並進加速度を求め、ジャイロ角速度・
     角加速度を使って運動学チェーンで各リンクIMU位置の加速度を予測。
     予測比力と実測比力の残差は相対角(=たわみdelta)に鋭敏なので、
     deltaの絶対値を強く拘束できる。ヤコビアンは数値微分で計算。
"""

import numpy as np

G = 9.81
G_VEC = np.array([0.0, -G])


def rot(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def perp(v):
    """2Dの k×v"""
    return np.array([-v[1], v[0]])


class MultiImuESKF:
    NX = 6  # [theta0, d1, d2, b0, b1, b2]

    def __init__(self, dt, geom,
                 gyro_noise=0.02, enc_rate_noise=0.01,
                 bias_walk=2e-4,
                 incl_sigma=0.5, incl_dyn_weight=5.0,
                 link_acc_sigma=0.7,
                 alpha_lpf_hz=15.0):
        """
        geom: dict(r_hip=(2,), d1, d2, L1)
        incl_sigma      : 胴体傾斜観測の基本σ [rad](緩い更新)
        link_acc_sigma  : リンク比力残差のσ [m/s^2](角加速度ノイズ込み)
        """
        self.dt = dt
        self.geom = geom
        self.x = np.zeros(self.NX)
        self.P = np.diag([0.2, 0.1, 0.1, 0.05, 0.05, 0.05]) ** 2

        s2g = gyro_noise ** 2
        s2e = enc_rate_noise ** 2
        Qc = np.zeros((3, 3))
        Qc[0, 0] = s2g
        Qc[1, 1] = 2 * s2g + s2e
        Qc[2, 2] = 2 * s2g + s2e
        Qc[0, 1] = Qc[1, 0] = -s2g
        Qc[1, 2] = Qc[2, 1] = -s2g
        Q = np.zeros((self.NX, self.NX))
        Q[:3, :3] = Qc * dt ** 2
        Q[3:, 3:] = np.eye(3) * (bias_walk ** 2) * dt
        self.Q = Q

        self.incl_R0 = incl_sigma ** 2
        self.incl_dyn = incl_dyn_weight
        self.link_R = link_acc_sigma ** 2

        F = np.eye(self.NX)
        F[0, 3] = -dt
        F[1, 3] = +dt; F[1, 4] = -dt
        F[2, 4] = +dt; F[2, 5] = -dt
        self.F = F

        # 角加速度推定用: ジャイロのLPF + 差分
        self._lpf_a = dt / (dt + 1.0 / (2 * np.pi * alpha_lpf_hz))
        self._gyro_f = None
        self._gyro_f_prev = None

    # ------------------------------------------------------------------
    def predict(self, gyro, qm_dot):
        w0, w1, w2 = gyro - self.x[3:]
        dt = self.dt
        self.x[0] += w0 * dt
        self.x[1] += (w1 - w0 - qm_dot[0]) * dt
        self.x[2] += (w2 - w1 - qm_dot[1]) * dt
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    def _alpha(self, gyro):
        """因果LPF付きジャイロ差分による角加速度推定"""
        if self._gyro_f is None:
            self._gyro_f = gyro.copy()
            self._gyro_f_prev = gyro.copy()
            return np.zeros(3)
        self._gyro_f_prev = self._gyro_f.copy()
        self._gyro_f = self._gyro_f + self._lpf_a * (gyro - self._gyro_f)
        return (self._gyro_f - self._gyro_f_prev) / self.dt

    # ------------------------------------------------------------------
    def _predict_link_forces(self, x, f0, gyro, alpha, qm):
        """状態xの下でリンク1,2のIMU比力を運動学で予測 (4,)"""
        th0, d1v, d2v, b0, b1, b2 = x
        w0, w1, w2 = gyro - np.array([b0, b1, b2])
        a0, a1, a2 = alpha
        gm = self.geom
        phi0 = th0
        phi1 = th0 + qm[0] + d1v
        phi2 = phi1 + qm[1] + d2v

        # 胴体点の並進加速度(胴体加速度計から, 恒等式)
        a_body = rot(phi0) @ f0 + G_VEC
        # 股関節点
        r_h = rot(phi0) @ gm['r_hip']
        a_hip = a_body + a0 * perp(r_h) - w0 ** 2 * r_h
        # 大腿IMU
        e1 = np.array([np.cos(phi1), np.sin(phi1)])
        p1 = gm['d1'] * e1
        a_i1 = a_hip + a1 * perp(p1) - w1 ** 2 * p1
        f1 = rot(phi1).T @ (a_i1 - G_VEC)
        # 膝点 -> 下腿IMU
        pk = gm['L1'] * e1
        a_knee = a_hip + a1 * perp(pk) - w1 ** 2 * pk
        e2 = np.array([np.cos(phi2), np.sin(phi2)])
        p2 = gm['d2'] * e2
        a_i2 = a_knee + a2 * perp(p2) - w2 ** 2 * p2
        f2 = rot(phi2).T @ (a_i2 - G_VEC)
        return np.concatenate([f1, f2])

    # ------------------------------------------------------------------
    def update_link_acc(self, f0, f_links, gyro, qm, alpha=None):
        """リンクIMU比力残差更新(数値ヤコビアン)
        alpha: 角加速度(3,)。Noneなら内部の因果LPF差分で推定。
        オフライン/固定ラグ許容ならゼロ位相平滑化した値を渡すと高精度。"""
        if alpha is None:
            alpha = self._alpha(gyro)
        z = np.concatenate([f_links[0], f_links[1]])
        h0 = self._predict_link_forces(self.x, f0, gyro, alpha, qm)

        H = np.zeros((4, self.NX))
        eps = 1e-6
        for k in range(self.NX):
            xp = self.x.copy()
            xp[k] += eps
            H[:, k] = (self._predict_link_forces(xp, f0, gyro, alpha, qm) - h0) / eps

        y = z - h0
        R = np.eye(4) * self.link_R
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(4))
        self.x = self.x + K @ y
        I_KH = np.eye(self.NX) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # ------------------------------------------------------------------
    def update_body_incl(self, f0):
        """胴体傾斜の緩い更新(theta0/b0ドリフト抑制)"""
        z = np.arctan2(f0[0], f0[1])
        h = self.x[0]
        y = np.arctan2(np.sin(z - h), np.cos(z - h))
        R = self.incl_R0 + self.incl_dyn * ((np.linalg.norm(f0) - G) / G) ** 2
        H = np.zeros((1, self.NX)); H[0, 0] = 1.0
        S = float((H @ self.P @ H.T).item()) + R
        K = (self.P @ H.T / S).ravel()
        self.x = self.x + K * y
        I_KH = np.eye(self.NX) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R

    # ------------------------------------------------------------------
    def step(self, gyro, acc, qm, qm_dot, alpha=None):
        self.predict(gyro, qm_dot)
        self.update_body_incl(acc[0])
        self.update_link_acc(acc[0], (acc[1], acc[2]), gyro, qm, alpha)
        return self.x.copy()


def smooth_alpha(gyro, dt, hz=30.0):
    """ゼロ位相(前進+後退)指数平滑 -> 中央差分で角加速度を推定。
    オンライン実装では同カットオフのFIR微分器(固定ラグ数ms)で代替可能。"""
    a = dt / (dt + 1.0 / (2 * np.pi * hz))
    g = gyro.copy()
    for i in range(1, len(g)):
        g[i] = g[i-1] + a * (g[i] - g[i-1])
    for i in range(len(g) - 2, -1, -1):
        g[i] = g[i+1] + a * (g[i] - g[i+1])
    return np.gradient(g, dt, axis=0)


def run_estimator(data, geom, use_smoothed_alpha=True, link_acc_sigma=0.25):
    n = len(data['t'])
    ekf = MultiImuESKF(dt=data['dt'], geom=geom, link_acc_sigma=link_acc_sigma)
    alphas = (smooth_alpha(data['gyro'], data['dt'])
              if use_smoothed_alpha else [None] * n)
    xs = np.zeros((n, MultiImuESKF.NX))
    for i in range(n):
        xs[i] = ekf.step(data['gyro'][i], data['acc'][i],
                         data['qm'][i], data['qm_dot'][i], alphas[i])
    return xs
