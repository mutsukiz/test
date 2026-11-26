"""
観測モデル整合性チェック
========================
シミュレータ(またはロギングした実機データ)の真値状態で推定器の
観測モデル _h_leg を評価し、実測との残差を表示する。
座標系・符号・取り付け位置の食い違いを推定を回す前に検出できる。

使い方:
  python check_model.py              # numpyバックエンド
  python check_model.py --pybullet   # PyBulletバックエンド
残差の目安(99パーセンタイル): gyro < 0.05 rad/s, acc < 0.3 m/s^2, ToF < 0.02 m なら整合。\nたわみ角加速度 δ̈ も真値(δの2階差分)を入れて評価する。
それ以上ならジオメトリ定義(quad_geom.py)とバックエンドの
リンク座標系が食い違っている。
"""

import sys
import numpy as np
from estimator3d import MultiImuESKF3D, State, smooth_zero_phase


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
    if hasattr(sim, 'acc_range'):
        sim.acc_range = np.inf
        sim.gyro_range = np.inf
    data = sim.run()

    dt = data['dt']
    ekf = MultiImuESKF3D(dt=dt)
    alpha_B = np.gradient(smooth_zero_phase(data['gyro'][:, 0], dt, 60),
                          dt, axis=0)
    qm_ddot = np.gradient(smooth_zero_phase(data['qm_dot'], dt, 60),
                          dt, axis=0)
    ddot_true = np.gradient(data['delta_true'], dt, axis=0)
    dddot_true = np.gradient(ddot_true, dt, axis=0)

    def state_at(i):
        return State(q=data['quat_true'][i], delta=data['delta_true'][i],
                     ddot=ddot_true[i], dddot=dddot_true[i],
                     p=data['pos_true'][i])

    n = len(data['t'])
    # 接地の切り替わり(着地・離地)の前後30 msは判定から外す。衝撃は数百 Hzの成分を持ち、
    # 500 Hzのデータでは真値側(δ̈ = δの2階差分)も追えないため。
    near_event = np.zeros(n, bool)
    if 'contact' in data:
        c = data['contact'].astype(int)
        w = int(round(0.03 / dt))
        for k in np.where(np.abs(np.diff(c, axis=0)).sum(1) > 0)[0]:
            near_event[max(0, k - w):k + w + 1] = True
    errs_g, errs_a, errs_ev = [], [], []
    for i in range(int(0.3 * n), int(0.9 * n), 5):
        x = state_at(i)
        ekf._w_B_input = data['gyro'][i, 0]
        for leg in range(4):
            h = ekf._h_leg(x, leg, data['acc'][i, 0], alpha_B[i],
                           data['qm'][i], data['qm_dot'][i], qm_ddot[i])
            zg = np.concatenate([data['gyro'][i, 1 + 2*leg],
                                 data['gyro'][i, 2 + 2*leg]])
            za = np.concatenate([data['acc'][i, 1 + 2*leg],
                                 data['acc'][i, 2 + 2*leg]])
            if near_event[i]:
                errs_ev.append(za - h[6:])
                continue
            errs_g.append(zg - h[:6])
            errs_a.append(za - h[6:])

    # 判定は99パーセンタイルで行う(PyBulletでは着地の衝撃が500 Hzで分解できず、
    # 最大値は数サンプルのスパイクで決まるため)。最大値は参考として表示する。
    eg = np.abs(np.array(errs_g))
    ea = np.abs(np.array(errs_a))
    print(f'gyro model |err| p99/max/mean [rad/s]: '
          f'{np.percentile(eg, 99):.5f} / {eg.max():.5f} / {eg.mean():.5f}')
    print(f'acc  model |err| p99/max/mean [m/s2] : '
          f'{np.percentile(ea, 99):.4f} / {ea.max():.4f} / {ea.mean():.4f}')
    ok = np.percentile(eg, 99) < 0.05 and np.percentile(ea, 99) < 0.3
    if errs_ev:
        ee = np.abs(np.array(errs_ev))
        print(f'  (参考) 接地の切り替わり±30 ms の加速度残差 p99/max: '
              f'{np.percentile(ee, 99):.2f} / {ee.max():.2f} m/s2'
              f'  [全体の{len(errs_ev) / (len(errs_ev) + len(errs_a)):.0%}]')

    # --- ToF観測モデルの整合性(データにtofがあれば) ---
    if 'tof' in data:
        errs_t = []
        for i in range(int(0.3 * n), int(0.9 * n), 5):
            x = state_at(i)
            for leg in range(4):
                z = data['tof'][i, leg]
                if not np.isfinite(z):
                    continue
                h = ekf._h_tof(x, leg, data['qm'][i])
                if h is not None:
                    errs_t.append(z - h)
        if errs_t:
            et = np.abs(np.array(errs_t))
            print(f'tof  model |err| p99/max/mean [m]    : '
                  f'{np.percentile(et, 99):.4f} / {et.max():.4f} / {et.mean():.4f}')
            ok = ok and np.percentile(et, 99) < 0.02
        else:
            print('tof  model: 有効な観測なし(接地判定・レンジを確認)')

    print('=> 整合' if ok else '=> 不整合: quad_geom.pyとバックエンドの座標系を確認')


if __name__ == '__main__':
    main()
