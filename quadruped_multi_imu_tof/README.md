# quadruped_multi_imu_tof

四足ロボットの胴体・大腿・下腿に計9個のIMU、足裏に4個のToF距離センサを付け、
Error-State EKFで「胴体姿勢・胴体位置・全12関節のたわみ(リンク角−モータ角)」を
同時推定するPythonプロトタイプ。

- `multi_imu_leg/`     : 平面2リンク脚の2Dプロトタイプ(手法の検証用、最小構成)
- `multi_imu_quad3d/`  : 3D四足・9 IMU+足裏ToF版(本体)

詳細な設計・検証結果・引継ぎ事項は別途の「引継書」を参照。

## クイックスタート

    pip install -r requirements.txt        # pybulletは任意
    cd multi_imu_quad3d
    python check_model.py                  # 観測モデル整合性チェック(数秒)
    python demo3d.py                       # 推定+評価+result3d.png(数分)

    cd ../multi_imu_leg
    python demo.py                         # 2D版(数十秒)

PyBulletを使う場合は `--pybullet` を付ける(未検証、必ずcheck_modelから)。
