"""
四足ロボット共有ジオメトリ定義
==============================
座標系: 胴体 x前方, y左, z上。重力 g = [0,0,-G]。
各脚: 股外転(roll, 軸x) -> 股ピッチ(軸y) -> 膝ピッチ(軸y)。
大腿・下腿はリンク座標の -z 方向に伸びる。

IMU配置(計9個):
  index 0        : 胴体(胴体座標原点)
  index 1+2*leg  : 大腿(股関節から d1 下方)
  index 2+2*leg  : 下腿(膝関節から d2 下方)
脚順: 0=FL, 1=FR, 2=RL, 3=RR
関節ベクトル(12,): [FL(1,2,3), FR(1,2,3), RL(1,2,3), RR(1,2,3)]
"""

import numpy as np

G = 9.81
G_VEC = np.array([0.0, 0.0, -G])

LEG_NAMES = ['FL', 'FR', 'RL', 'RR']

HIP_OFFSETS = np.array([
    [+0.15, +0.08, 0.0],   # FL
    [+0.15, -0.08, 0.0],   # FR
    [-0.15, +0.08, 0.0],   # RL
    [-0.15, -0.08, 0.0],   # RR
])

JOINT_AXES = [np.array([1.0, 0, 0]),   # 股外転 (roll)
              np.array([0, 1.0, 0]),   # 股ピッチ
              np.array([0, 1.0, 0])]   # 膝ピッチ

L1 = 0.20   # 大腿長 [m]
L2 = 0.20   # 下腿長 [m]
D1 = 0.10   # 大腿IMU位置(股関節から)
D2 = 0.10   # 下腿IMU位置(膝関節から)

DOWN = np.array([0.0, 0.0, -1.0])   # リンク伸長方向(リンク座標)


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def leg_rotations(q3):
    """関節角(3,) -> (R1, R12, R123)  股外転/大腿/下腿の胴体からの相対回転"""
    R1 = rx(q3[0])
    R12 = R1 @ ry(q3[1])
    R123 = R12 @ ry(q3[2])
    return R1, R12, R123


# ---------------------------------------------------------------- foot
FOOT_R = 0.02   # 足先の球の半径 [m]。球の中心は膝から L2 下方(下腿座標)

# ---------------------------------------------------------------- ToF sensor
# 足裏ToF (ST VL53L4CD, DS13812 Rev 8)。
# 足先の球の横(下腿座標 ±y、外側)に取り付け、下腿が鉛直で接地したとき
# 地面から TOF_CLEARANCE だけ浮く高さに置く。測距方向は下腿座標 -z。
#  - 横オフセットはピッチ回転(下腿y軸まわり)で高さが変わらないように y 方向に取る。
#  - 接地中の測距値は TOF_CLEARANCE/|u_z| 程度(2〜5 mm)で、線形域(1 mm〜)に入る。
#  - 球(半径20 mm)は高さ2 mmの位置で半径約9 mmなので、12 mm横に置けば足は視野に入らない。
TOF_CLEARANCE = 0.002        # 接地時(下腿鉛直)のセンサ面の地上高 [m]
TOF_LATERAL = 0.012          # 球中心軸からの横オフセット [m](外側)
_LEG_SIDE = np.array([+1.0, -1.0, +1.0, -1.0])   # FL, FR, RL, RR の外側(+y が左)
TOF_MOUNTS = np.array([[0.0, s * TOF_LATERAL, -(L2 + FOOT_R - TOF_CLEARANCE)]
                       for s in _LEG_SIDE])      # 取り付け位置(下腿座標, 膝から) (4,3)
TOF_RAY = DOWN               # 測距方向(下腿座標)
# 誤差: DS表17の「1〜100 mmで±6〜7 mm」は個体差・オフセット・ノイズ込みの90%範囲。
# オフセット校正後の測定ごとのばらつき(TOF_SIGMA)と、校正残りのセンサ固有オフセット
# (TOF_OFFSET_MAG、一様乱数の幅)に分けて扱う。どちらも仮定値なので実測で置き換えること。
TOF_SIGMA = 0.002            # 測定ごとのノイズ 1σ [m](仮定)
TOF_OFFSET_MAG = 0.0         # センサ固有オフセットの幅 [m](既定0。例: 0.003)
TOF_RANGE = (0.0, 0.45)      # 出力範囲 [m]。DS 6.2: 検出は0 mmから、線形応答は1 mmから。
                             # 上限はDS 6.3の灰17%・検出率90%で約450 mm
TOF_MAX_TILT = 0.5           # ray世界z成分がこの値より浅い(>-0.5)と無効
TOF_RATE_HZ = 50.0           # 測定レート [Hz](DS: 最大100 Hz)
TOF_BUDGET = 0.02            # タイミングバジェット [s]。測定値はこの区間の平均距離


def tof_measure(d_true, dt, rng, sigma=TOF_SIGMA, budget=TOF_BUDGET,
                offset_mag=TOF_OFFSET_MAG, rng_range=TOF_RANGE):
    """真の距離の時系列(n,4)(無効はNaN)から、ToFの測定値(n,4)を合成する。
    各時刻の値は直前 budget 秒間の平均距離 + オフセット + ノイズ
    (区間内に無効があれば無効)。センサは負の距離を出さないので0で切る。
    推定器はこれを測定時刻(=区間の終わり)の値として使うので、遊脚中は
    約 budget/2 の遅れが誤差として現れる。"""
    n = d_true.shape[0]
    w = max(1, int(round(budget / dt)))
    c = np.nancumsum(np.r_[np.zeros((1, d_true.shape[1])), d_true], axis=0)
    bad = np.cumsum(np.r_[np.zeros((1, d_true.shape[1])), ~np.isfinite(d_true)], axis=0)
    out = np.full_like(d_true, np.nan)
    for i in range(w - 1, n):
        ok = (bad[i + 1] - bad[i + 1 - w]) == 0
        out[i, ok] = (c[i + 1, ok] - c[i + 1 - w, ok]) / w
    offset = rng.uniform(-1, 1, size=d_true.shape[1]) * offset_mag
    out += offset[None] + sigma * rng.standard_normal(out.shape)
    out[out > rng_range[1]] = np.nan
    out = np.maximum(out, rng_range[0])     # NaNはそのまま残る
    return out
