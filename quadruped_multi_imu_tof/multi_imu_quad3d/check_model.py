"""
観測モデル整合性チェック
========================
シミュレータ(またはロギングした実機データ)の真値状態で推定器の
観測モデル _h_leg を評価し、実測との残差を表示する。
座標系・符号・取り付け位置の食い違いを推定を回す前に検出できる。

使い方:
  python check_model.py              # numpyバックエンド
  python check_model.py --pybullet   # PyBulletバックエンド
残差の目安: gyro < 0.05 rad/s, acc < 0.3 m/s^2 なら整合。
それ以上ならジオメトリ定義(quad_geom.py)とバックエンドの
リンク座標系が食い違っている。
"""

import sys
import numpy as np
from estimator3d import MultiImuESKF3D, smooth_zero_phase


def main():
    if '--pybullet' in sys.argv:
        from sim3d_pybullet import QuadSimulatorPyBullet
        sim = QuadSimulatorPyBullet(duration=3.0)
    else:
        from sim3d_numpy import QuadSimulator3D
        sim = QuadSimulator3D(dt=0.002, duration=3.0, seed=1)
    # ノイズ・バイアス・ToFの平均化遅れを切り、幾何と運動学だけを比べる
    sim.gyro_noise = 0
    sim.acc_noise = 0
    sim.enc_rate_noise = 0
    sim.enc_quant = 1e-12
    sim.gyro_bias_true[:] = 0
    sim.tof_sigma = 0
    sim.tof_budget = 0
    sim.tof_offset_mag = 0
    data = sim.run()

    dt = data['dt']
    ekf = MultiImuESKF3D(dt=dt)
    alpha_B = np.gradient(smooth_zero_phase(data['gyro'][:, 0], dt, 60),
                          dt, axis=0)
    qm_ddot = np.gradient(smooth_zero_phase(data['qm_dot'], dt, 60),
                          dt, axis=0)
    ddot_true = np.gradient(data['delta_true'], dt, axis=0)

    n = len(data['t'])
    errs_g, errs_a = [], []
    for i in range(int(0.3 * n), int(0.9 * n), 25):
        state = (data['quat_true'][i], data['delta_true'][i],
                 ddot_true[i], np.zeros((9, 3)),
                 data['pos_true'][i], np.zeros(3), np.zeros((9, 3)))
        ekf._w_B_input = data['gyro'][i, 0]
        for leg in range(4):
            h = ekf._h_leg(state, leg, data['acc'][i, 0], alpha_B[i],
                           data['qm'][i], data['qm_dot'][i], qm_ddot[i])
            zg = np.concatenate([data['gyro'][i, 1 + 2*leg],
                                 data['gyro'][i, 2 + 2*leg]])
            za = np.concatenate([data['acc'][i, 1 + 2*leg],
                                 data['acc'][i, 2 + 2*leg]])
            errs_g.append(zg - h[:6])
            errs_a.append(za - h[6:])

    eg = np.abs(np.array(errs_g))
    ea = np.abs(np.array(errs_a))
    print(f'gyro model |err| max/mean [rad/s]: {eg.max():.5f} / {eg.mean():.5f}')
    print(f'acc  model |err| max/mean [m/s2] : {ea.max():.4f} / {ea.mean():.4f}')
    ok = eg.max() < 0.05 and ea.max() < 0.3

    # --- ToF観測モデルの整合性(データにtofがあれば) ---
    if 'tof' in data:
        errs_t = []
        for i in range(int(0.3 * n), int(0.9 * n), 25):
            state = (data['quat_true'][i], data['delta_true'][i],
                     ddot_true[i], np.zeros((9, 3)),
                     data['pos_true'][i], np.zeros(3), np.zeros((9, 3)))
            for leg in range(4):
                z = data['tof'][i, leg]
                if not np.isfinite(z):
                    continue
                h = ekf._h_tof(state, leg, data['qm'][i])
                if h is not None:
                    errs_t.append(z - h)
        if errs_t:
            et = np.abs(np.array(errs_t))
            print(f'tof  model |err| max/mean [m]    : {et.max():.4f} / {et.mean():.4f}')
            ok = ok and et.max() < 0.02
        else:
            print('tof  model: 有効な観測なし(接地判定・レンジを確認)')

    print('=> 整合' if ok else '=> 不整合: quad_geom.pyとバックエンドの座標系を確認')


if __name__ == '__main__':
    main()
