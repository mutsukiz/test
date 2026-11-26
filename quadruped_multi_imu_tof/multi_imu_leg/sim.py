"""
平面2リンク脚シミュレータ(合成データ生成)
========================================
構成:
  - 胴体: ピッチ角 theta0(浮遊, 並進も微小振動)
  - 関節1(股), 関節2(膝): モータ角 q_m(エンコーダで計測)と
    リンク角 q = q_m + delta(deltaは関節弾性によるたわみ)
  - IMU x3: 胴体 / 大腿リンク / 下腿リンク(ジャイロ+加速度計, バイアス・ノイズ付き)

真値のたわみ delta は、重力・慣性負荷トルクに対する
2次系(ばね-ダンパ)応答として生成する。
"""

import numpy as np

G = 9.81


def rot(phi):
    """2D回転行列(world <- link)"""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


class PlanarLegSimulator:
    def __init__(self, dt=0.002, duration=10.0, seed=0):
        self.dt = dt
        self.duration = duration
        self.rng = np.random.default_rng(seed)

        # --- 機構パラメータ ---
        self.L1 = 0.20          # 大腿長 [m]
        self.L2 = 0.20          # 下腿長 [m]
        self.lc1, self.lc2 = 0.10, 0.10   # 重心位置
        self.m1, self.m2 = 1.0, 0.5       # リンク質量 [kg]
        self.r_hip = np.array([0.15, -0.05])  # 胴体原点->股関節(胴体座標)
        self.d1, self.d2 = 0.12, 0.12    # IMU取り付け位置(関節から)[m]

        # --- 関節弾性(たわみ)パラメータ ---
        self.k_joint = np.array([60.0, 40.0])   # 剛性 [Nm/rad](低剛性を想定)
        self.wn = 25.0    # たわみ動特性の固有角周波数 [rad/s]
        self.zeta = 0.7

        # --- センサノイズ ---
        self.gyro_noise = 0.02        # [rad/s] (1σ, サンプル毎)
        self.acc_noise = 0.05         # [m/s^2]
        self.gyro_bias_true = np.array([0.02, -0.015, 0.03])  # 定常バイアス
        self.enc_quant = 0.001        # エンコーダ量子化 [rad]
        self.enc_rate_noise = 0.01    # モータ速度ノイズ [rad/s]

    def run(self):
        dt = self.dt
        t = np.arange(0.0, self.duration, dt)
        n = len(t)

        # --- 指令(モータ)軌道: トロット様の周期運動 ---
        f = 1.5  # 歩容周波数 [Hz]
        qm1 = 0.6 + 0.30 * np.sin(2 * np.pi * f * t)
        qm2 = -1.2 + 0.40 * np.sin(2 * np.pi * f * t + 1.0)
        qm = np.stack([qm1, qm2], axis=1)
        qm_dot = np.gradient(qm, dt, axis=0)
        qm_ddot = np.gradient(qm_dot, dt, axis=0)

        # --- 胴体運動(真値) ---
        theta0 = 0.10 * np.sin(2 * np.pi * 0.5 * t) \
               + 0.03 * np.sin(2 * np.pi * 2.0 * t + 0.7)
        theta0_dot = np.gradient(theta0, dt)
        p_body = np.stack([0.05 * np.sin(2 * np.pi * f * t),
                           0.02 * np.sin(2 * np.pi * 2 * f * t)], axis=1)

        # --- たわみ真値: 負荷トルク -> 2次系応答 ---
        phi1_nom = theta0 + qm1
        phi2_nom = phi1_nom + qm2
        J1, J2 = 0.02, 0.01  # 反射慣性の目安
        tau1 = ((self.m1 * self.lc1 + self.m2 * self.L1) * G * np.cos(phi1_nom)
                + self.m2 * self.lc2 * G * np.cos(phi2_nom)
                + J1 * qm_ddot[:, 0])
        tau2 = self.m2 * self.lc2 * G * np.cos(phi2_nom) + J2 * qm_ddot[:, 1]
        delta_ss = -np.stack([tau1, tau2], axis=1) / self.k_joint

        delta = np.zeros((n, 2))
        delta_dot = np.zeros(2)
        d = delta_ss[0].copy()
        for i in range(n):
            dd = self.wn**2 * (delta_ss[i] - d) - 2 * self.zeta * self.wn * delta_dot
            delta_dot = delta_dot + dd * dt
            d = d + delta_dot * dt
            delta[i] = d

        # --- リンク角(真値) ---
        q = qm + delta
        phi1 = theta0 + q[:, 0]
        phi2 = phi1 + q[:, 1]
        phi = np.stack([theta0, phi1, phi2], axis=1)   # 各IMUの絶対角
        omega = np.gradient(phi, dt, axis=0)            # 角速度真値

        # --- IMU位置と加速度(数値微分) ---
        p_hip = p_body + np.einsum('nij,j->ni',
                                   np.stack([rot(a) for a in theta0]), self.r_hip)
        e1 = np.stack([np.cos(phi1), np.sin(phi1)], axis=1)
        e2 = np.stack([np.cos(phi2), np.sin(phi2)], axis=1)
        p_imu = np.stack([p_body,
                          p_hip + self.d1 * e1,
                          p_hip + self.L1 * e1 + self.d2 * e2], axis=1)  # (n,3,2)
        v_imu = np.gradient(p_imu, dt, axis=0)
        a_imu = np.gradient(v_imu, dt, axis=0)

        # --- センサ出力 ---
        g_vec = np.array([0.0, -G])
        gyro = (omega + self.gyro_bias_true
                + self.gyro_noise * self.rng.standard_normal((n, 3)))
        acc = np.zeros((n, 3, 2))
        for j in range(3):
            for i in range(n):
                acc[i, j] = rot(phi[i, j]).T @ (a_imu[i, j] - g_vec)
        acc += self.acc_noise * self.rng.standard_normal((n, 3, 2))

        qm_meas = np.round(qm / self.enc_quant) * self.enc_quant
        qm_dot_meas = qm_dot + self.enc_rate_noise * self.rng.standard_normal((n, 2))

        return {
            't': t, 'dt': dt,
            # 真値
            'theta0_true': theta0, 'delta_true': delta,
            'phi_true': phi, 'gyro_bias_true': self.gyro_bias_true,
            # 計測値
            'gyro': gyro, 'acc': acc,
            'qm': qm_meas, 'qm_dot': qm_dot_meas,
        }


if __name__ == '__main__':
    data = PlanarLegSimulator().run()
    print('generated:', {k: getattr(v, 'shape', v) for k, v in data.items()
                         if hasattr(v, 'shape')})
    print('max |delta| [deg]:', np.rad2deg(np.abs(data['delta_true']).max(axis=0)))
