"""
numpy製 3D四足シミュレータ(推定器検証用・運動学ベース)
========================================================
胴体の姿勢・並進とモータ軌道を規定し、関節たわみ真値を
負荷トルクに対する2次系応答として生成。全リンクの姿勢・IMU位置を
順運動学で計算し、数値微分で角速度・加速度の真値を得て、
ノイズ・バイアス付きのIMU出力(9個)とエンコーダ出力を合成する。

出力データ形式(PyBulletバックエンド sim3d_pybullet.py と共通):
  t (n,), dt
  gyro (n,9,3), acc (n,9,3)      # [胴体, FL大腿, FL下腿, FR..., RL..., RR...]
  qm (n,12), qm_dot (n,12)       # モータ角(エンコーダ)
  quat_true (n,4)  [w,x,y,z]     # 胴体姿勢真値
  delta_true (n,12)              # たわみ真値
  gyro_bias_true (9,3)
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from quad_geom import (G, G_VEC, HIP_OFFSETS, L1, L2, D1, D2, DOWN,
                       leg_rotations, TOF_MOUNT, TOF_RAY, TOF_SIGMA,
                       TOF_RANGE, TOF_MAX_TILT)

NL = 4
NJ = 12
NIMU = 9


class QuadSimulator3D:
    def __init__(self, dt=0.002, duration=10.0, seed=0, gait_hz=1.5,
                 accel_bias_mag=0.0):
        self.dt = dt
        self.gait_hz = gait_hz
        self.accel_bias_mag = accel_bias_mag
        self.duration = duration
        self.rng = np.random.default_rng(seed)

        # 関節剛性(低剛性想定)と тわみ動特性
        self.k_joint = np.array([80.0, 60.0, 40.0])   # [外転, 股P, 膝] Nm/rad
        self.wn, self.zeta = 25.0, 0.7

        # センサ
        self.gyro_noise = 0.02
        self.acc_noise = 0.05
        self.enc_quant = 0.001
        self.enc_rate_noise = 0.01
        self.gyro_bias_true = self.rng.uniform(-0.03, 0.03, size=(NIMU, 3))
        self.acc_bias_true = self.rng.uniform(
            -1, 1, size=(NIMU, 3)) * self.accel_bias_mag

    # ------------------------------------------------------------------
    def run(self):
        dt = self.dt
        t = np.arange(0.0, self.duration, dt)
        n = len(t)
        f = self.gait_hz  # 歩容周波数
        leg_phase = np.array([0.0, np.pi, np.pi, 0.0])  # トロット

        # --- モータ軌道 (n,12) ---
        qm = np.zeros((n, NJ))
        for leg in range(NL):
            ph = leg_phase[leg]
            j = 3 * leg
            qm[:, j+0] = 0.05 * np.sin(2*np.pi*f*t + ph + 0.5)
            qm[:, j+1] = 0.50 + 0.25 * np.sin(2*np.pi*f*t + ph)
            qm[:, j+2] = -1.00 + 0.35 * np.sin(2*np.pi*f*t + ph + 0.8)
        qm_dot = np.gradient(qm, dt, axis=0)
        qm_ddot = np.gradient(qm_dot, dt, axis=0)

        # --- 胴体運動(真値) ---
        rpy = np.stack([
            0.03 * np.sin(2*np.pi*1.5*t + 0.3),
            0.06 * np.sin(2*np.pi*0.75*t),
            0.02 * np.sin(2*np.pi*0.3*t)], axis=1)
        R_WB = Rot.from_euler('xyz', rpy)
        quat_scipy = R_WB.as_quat()  # (x,y,z,w)
        quat_true = np.column_stack([quat_scipy[:, 3], quat_scipy[:, :3]])
        p_B = np.stack([0.03 * np.sin(2*np.pi*f*t),
                        0.02 * np.sin(2*np.pi*f*t + 1.0),
                        0.30 + 0.015 * np.sin(2*np.pi*2*f*t)], axis=1)

        # --- たわみ真値: 負荷トルクproxy -> 2次系 ---
        delta = np.zeros((n, NJ))
        for leg in range(NL):
            j = 3 * leg
            ph = leg_phase[leg]
            load = 0.4 * np.sin(2*np.pi*f*t + ph + 0.3)   # 接触負荷のproxy
            tau = np.stack([
                0.5 * np.sin(qm[:, j]) + 0.3 * load,
                2.0 * np.sin(qm[:, j+1]) + 0.03 * qm_ddot[:, j+1] + load,
                0.8 * np.sin(qm[:, j+1] + qm[:, j+2])
                + 0.015 * qm_ddot[:, j+2] + 0.5 * load], axis=1)
            dss = -tau / self.k_joint
            d = dss[0].copy()
            ddt = np.zeros(3)
            for i in range(n):
                dd = self.wn**2 * (dss[i] - d) - 2*self.zeta*self.wn*ddt
                ddt += dd * dt
                d += ddt * dt
                delta[i, j:j+3] = d

        q = qm + delta   # リンク角真値

        # --- 順運動学: 各IMUの回転行列と位置 ---
        Rmats = np.zeros((n, NIMU, 3, 3))
        pos = np.zeros((n, NIMU, 3))
        p_foot = np.zeros((n, NL, 3))
        RB = R_WB.as_matrix()
        Rmats[:, 0] = RB
        pos[:, 0] = p_B
        for i in range(n):
            for leg in range(NL):
                j = 3 * leg
                R1, R12, R123 = leg_rotations(q[i, j:j+3])
                R_WT = RB[i] @ R12
                R_WS = RB[i] @ R123
                p_hip = p_B[i] + RB[i] @ HIP_OFFSETS[leg]
                Rmats[i, 1 + 2*leg] = R_WT
                Rmats[i, 2 + 2*leg] = R_WS
                pos[i, 1 + 2*leg] = p_hip + R_WT @ (D1 * DOWN)
                pos[i, 2 + 2*leg] = (p_hip + R_WT @ (L1 * DOWN)
                                     + R_WS @ (D2 * DOWN))
                p_foot[i, leg] = (p_hip + R_WT @ (L1 * DOWN)
                                  + R_WS @ TOF_MOUNT)

        # --- 接地面(z=0)に対する高さ調整: 最下点クリアランス1cm ---
        # (運動学シミュレータのため接触は解かず、足先が地面上を掃くよう
        #  全体を鉛直シフトする。定数シフトなので加速度は不変)
        shift = 0.01 - p_foot[:, :, 2].min()
        p_B[:, 2] += shift
        pos[:, :, 2] += shift
        p_foot[:, :, 2] += shift

        # --- 足裏ToF測距(地面z=0の平面, 有効範囲・角度ゲート) ---
        tof = np.full((n, NL), np.nan)
        for leg in range(NL):
            u = np.einsum('nij,j->ni', Rmats[:, 2 + 2*leg], TOF_RAY)
            valid = u[:, 2] < -TOF_MAX_TILT
            d = np.where(valid, -p_foot[:, leg, 2] / u[:, 2], np.nan)
            d += TOF_SIGMA * self.rng.standard_normal(n)
            d[(d < TOF_RANGE[0]) | (d > TOF_RANGE[1])] = np.nan
            tof[:, leg] = d

        # --- 角速度真値(中央差分, 回転行列の相対回転から) ---
        omega_W = np.zeros((n, NIMU, 3))
        for k in range(NIMU):
            Rk = Rmats[:, k]
            dR = np.einsum('nij,nkj->nik', Rk[2:], Rk[:-2])  # R_{t+1} R_{t-1}^T
            rv = Rot.from_matrix(dR).as_rotvec() / (2 * dt)
            omega_W[1:-1, k] = rv
            omega_W[0, k] = omega_W[1, k]
            omega_W[-1, k] = omega_W[-2, k]

        # --- 加速度真値(位置の2階数値微分) ---
        vel = np.gradient(pos, dt, axis=0)
        accel = np.gradient(vel, dt, axis=0)

        # --- センサ出力 ---
        gyro = np.zeros((n, NIMU, 3))
        acc = np.zeros((n, NIMU, 3))
        for i in range(n):
            for k in range(NIMU):
                Rk = Rmats[i, k]
                gyro[i, k] = Rk.T @ omega_W[i, k]
                acc[i, k] = Rk.T @ (accel[i, k] - G_VEC)
        gyro += self.gyro_bias_true[None, :, :]
        gyro += self.gyro_noise * self.rng.standard_normal((n, NIMU, 3))
        acc += self.acc_bias_true[None]
        acc += self.acc_noise * self.rng.standard_normal((n, NIMU, 3))

        qm_meas = np.round(qm / self.enc_quant) * self.enc_quant
        qm_dot_meas = qm_dot + self.enc_rate_noise * self.rng.standard_normal((n, NJ))

        return {
            't': t, 'dt': dt,
            'gyro': gyro, 'acc': acc,
            'qm': qm_meas, 'qm_dot': qm_dot_meas,
            'quat_true': quat_true, 'delta_true': delta,
            'gyro_bias_true': self.gyro_bias_true,
            'tof': tof, 'pos_true': p_B,
        }


if __name__ == '__main__':
    data = QuadSimulator3D(duration=2.0).run()
    print({k: getattr(v, 'shape', v) for k, v in data.items()
           if hasattr(v, 'shape')})
    print('max |delta| [deg]:', np.rad2deg(np.abs(data['delta_true']).max()))
