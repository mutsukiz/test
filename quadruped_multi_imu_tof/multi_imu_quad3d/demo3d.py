"""
3Dデモ: シミュレーション -> 9-IMU ESKF -> 評価・可視化
使い方:
  python demo3d.py                # numpyバックエンド(検証済み)
  python demo3d.py --pybullet     # PyBulletバックエンド(要 pip install pybullet)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from estimator3d import run_estimator3d

JOINT_LABELS = [f'{leg}-{j}' for leg in ['FL', 'FR', 'RL', 'RR']
                for j in ['abd', 'hip', 'knee']]


def main():
    if '--pybullet' in sys.argv:
        from sim3d_pybullet import QuadSimulatorPyBullet
        data = QuadSimulatorPyBullet(duration=10.0).run()
    else:
        from sim3d_numpy import QuadSimulator3D
        data = QuadSimulator3D(dt=0.002, duration=10.0, seed=42).run()

    has_tof = 'tof' in data
    res = run_estimator3d(data, use_tof=has_tof)
    res_notof = run_estimator3d(data, use_tof=False) if has_tof else res
    t = data['t']
    skip = int(1.0 / data['dt'])
    deg = np.rad2deg

    # --- 姿勢評価 ---
    q_e, q_t = res['quat'], data['quat_true']
    r_est = Rot.from_quat(np.column_stack([q_e[:, 1:], q_e[:, 0]]))
    r_true = Rot.from_quat(np.column_stack([q_t[:, 1:], q_t[:, 0]]))
    e_rpy = (r_true.inv() * r_est).as_euler('xyz')
    rpy_est = r_est.as_euler('xyz')
    rpy_true = r_true.as_euler('xyz')
    att_rmse = deg(np.sqrt(np.mean(e_rpy[skip:] ** 2, axis=0)))
    print('=== RMSE (t > 1s) ===')
    print(f'attitude roll/pitch/yaw [deg]: '
          f'{att_rmse[0]:.2f} / {att_rmse[1]:.2f} / {att_rmse[2]:.2f} '
          f'(yawは磁気/視覚なしのため原理的に不可観測でドリフト)')

    # --- たわみ評価 ---
    d_e, d_t = res['delta'], data['delta_true']
    rm = np.sqrt(np.mean((d_e[skip:] - d_t[skip:]) ** 2, axis=0))
    rm0 = np.sqrt(np.mean(d_t[skip:] ** 2, axis=0))
    for i, lab in enumerate(JOINT_LABELS):
        print(f'  delta {lab:8s}: est {deg(rm[i]):.3f} deg  '
              f'(rigid: {deg(rm0[i]):.3f} deg, x{rm0[i]/rm[i]:.1f})')
    if 'gyro_bias_true' in data:
        be = np.sqrt(np.mean((res['bias'][-1] - data['gyro_bias_true']) ** 2))
        print(f'gyro bias RMS error: {be:.4f} rad/s')
    if has_tof and 'pos_true' in data:
        ez = np.sqrt(np.mean((res['pos'][skip:, 2] - data['pos_true'][skip:, 2]) ** 2))
        ez0 = np.sqrt(np.mean((res_notof['pos'][skip:, 2] - data['pos_true'][skip:, 2]) ** 2))
        print(f'body height RMSE: ToFあり {ez*1000:.1f} mm  '
              f'/ ToFなし(INS単独ドリフト) {ez0*1000:.1f} mm')

    # --- 可視化 ---
    n_ax = 5 if (has_tof and 'pos_true' in data) else 4
    fig, axes = plt.subplots(n_ax, 1, figsize=(11, 3.1 * n_ax))

    ax = axes[0]
    for k, name, c in [(0, 'roll', 'C0'), (1, 'pitch', 'C1')]:
        ax.plot(t, deg(rpy_true[:, k]), c, ls='--', lw=1.2)
        ax.plot(t, deg(rpy_est[:, k]), c, lw=1.2, label=f'{name} est')
    ax.set_ylabel('attitude [deg]')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title('Body attitude (dashed = true)')

    for row, (jj, name) in enumerate([(1, 'FL hip pitch'), (2, 'FL knee')]):
        ax = axes[1 + row]
        ax.plot(t, deg(d_t[:, jj]), 'k', lw=1.4, label='true')
        ax.plot(t, deg(d_e[:, jj]), 'C0', lw=1.1, label='estimated')
        ax.axhline(0, color='C3', ls='--', lw=1, label='rigid assumption')
        ax.set_ylabel(f'{name}\ndeflection [deg]')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title('Joint deflection (compliance error)')

    ax = axes[3]
    x = np.arange(12)
    ax.bar(x - 0.2, deg(rm0), width=0.4, color='C3', label='rigid assumption')
    ax.bar(x + 0.2, deg(rm), width=0.4, color='C0', label='multi-IMU ESKF')
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, rotation=45, fontsize=8)
    ax.set_ylabel('deflection RMSE [deg]')
    ax.set_xlabel('joint')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('Deflection error: rigid model vs multi-IMU estimation')

    if has_tof and 'pos_true' in data:
        ax = axes[4]
        ax.plot(t, data['pos_true'][:, 2] * 1000, 'k', lw=1.4, label='true')
        ax.plot(t, res_notof['pos'][:, 2] * 1000, 'C3', lw=1.0,
               label='without ToF (INS drift)')
        ax.plot(t, res['pos'][:, 2] * 1000, 'C0', lw=1.2, label='with ToF')
        ax.set_ylabel('body height [mm]')
        ax.set_xlabel('time [s]')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_title('Body height: effect of foot ToF range sensors')

    fig.tight_layout()
    out = 'result3d_pybullet.png' if '--pybullet' in sys.argv else 'result3d.png'
    fig.savefig(out, dpi=130)
    print('saved:', out)


if __name__ == '__main__':
    main()
