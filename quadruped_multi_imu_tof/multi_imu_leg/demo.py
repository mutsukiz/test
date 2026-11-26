"""
デモ: シミュレーション -> マルチIMU ESKF推定 -> 評価・可視化
比較対象(ベースライン):
  - 胴体姿勢: 胴体IMUのみの相補フィルタ相当(ジャイロ積分+加速度傾斜)
  - たわみ: 剛体仮定(delta = 0)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sim import PlanarLegSimulator, G
from estimator import run_estimator


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    sim = PlanarLegSimulator(dt=0.002, duration=10.0, seed=42)
    data = sim.run()
    t = data['t']
    geom = dict(r_hip=sim.r_hip, d1=sim.d1, d2=sim.d2, L1=sim.L1)

    # --- 提案手法: マルチIMU ESKF ---
    xs = run_estimator(data, geom)
    th_est = xs[:, 0]
    delta_est = xs[:, 1:3]
    bias_est = xs[:, 3:6]

    # --- ベースライン: 胴体IMUのみ(単一IMU相補フィルタ相当) ---
    n = len(t)
    th_single = np.zeros(n)
    x = np.zeros(2)          # [theta0, b0]
    P = np.diag([0.1, 0.05]) ** 2
    Q = np.diag([sim.gyro_noise ** 2 * data['dt'] ** 2, (2e-4) ** 2 * data['dt']])
    for i in range(n):
        w0 = data['gyro'][i, 0] - x[1]
        x[0] += w0 * data['dt']
        F = np.array([[1, -data['dt']], [0, 1]])
        P = F @ P @ F.T + Q
        f = data['acc'][i, 0]
        z = np.arctan2(f[0], f[1])
        R = 0.5 ** 2 + 5.0 * ((np.linalg.norm(f) - G) / G) ** 2
        H = np.array([[1.0, 0.0]])
        S = float((H @ P @ H.T).item()) + R
        K = (P @ H.T / S).ravel()
        y = np.arctan2(np.sin(z - x[0]), np.cos(z - x[0]))
        x += K * y
        P = (np.eye(2) - np.outer(K, H)) @ P
        th_single[i] = x[0]

    # --- 評価 ---
    th_true = data['theta0_true']
    d_true = data['delta_true']
    skip = int(1.0 / data['dt'])  # 収束後1s以降で評価
    print('=== RMSE (t > 1s) ===')
    print(f'body pitch  multi-IMU : {np.rad2deg(rmse(th_est[skip:], th_true[skip:])):.3f} deg')
    print(f'body pitch  single-IMU: {np.rad2deg(rmse(th_single[skip:], th_true[skip:])):.3f} deg')
    for j in range(2):
        e = rmse(delta_est[skip:, j], d_true[skip:, j])
        e0 = rmse(np.zeros_like(d_true[skip:, j]), d_true[skip:, j])
        print(f'delta{j+1}  est: {np.rad2deg(e):.3f} deg   '
              f'(rigid assumption: {np.rad2deg(e0):.3f} deg)')
    print('gyro bias true:', data['gyro_bias_true'])
    print('gyro bias est :', bias_est[-1].round(4))

    # --- 可視化 ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    deg = np.rad2deg

    ax = axes[0]
    ax.plot(t, deg(th_true), 'k', lw=1.5, label='true')
    ax.plot(t, deg(th_single), 'C1', lw=1, alpha=0.8, label='single body IMU')
    ax.plot(t, deg(th_est), 'C0', lw=1, label='multi-IMU ESKF')
    ax.set_ylabel('body pitch [deg]')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    ax.set_title('Body pitch estimation')

    for j in range(2):
        ax = axes[1 + j]
        ax.plot(t, deg(d_true[:, j]), 'k', lw=1.5, label='true')
        ax.plot(t, deg(delta_est[:, j]), 'C0', lw=1, label='estimated')
        ax.axhline(0, color='C3', ls='--', lw=1, label='rigid assumption')
        ax.set_ylabel(f'joint {j+1} deflection [deg]')
        ax.legend(loc='upper right'); ax.grid(alpha=0.3)
        if j == 0:
            ax.set_title('Joint deflection (compliance error) estimation')

    ax = axes[3]
    for j in range(3):
        ax.plot(t, deg(bias_est[:, j]), f'C{j}', lw=1, label=f'IMU{j} est')
        ax.axhline(deg(data['gyro_bias_true'][j]), color=f'C{j}', ls='--', lw=1)
    ax.set_ylabel('gyro bias [deg/s]'); ax.set_xlabel('time [s]')
    ax.legend(loc='upper right', ncol=3); ax.grid(alpha=0.3)
    ax.set_title('Gyro bias estimation (dashed = true)')

    fig.tight_layout()
    fig.savefig('result.png', dpi=130)
    print('saved: result.png')


if __name__ == '__main__':
    main()
