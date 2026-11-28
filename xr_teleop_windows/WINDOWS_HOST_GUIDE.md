# xr_teleoperate:ホスト PC を Windows 機で代替するための手順書

Unitree G1 と Meta Quest 3 / 3S でテレオペレーションを行う構成のうち、**ホスト PC(`teleop_hand_and_arm.py` を動かす PC)を Windows 機で代替する**ための手順です。xr_teleoperate が公式に動作確認しているのは Ubuntu 20.04 / 22.04 だけなので、Windows で動かすにはこの手順書のパッチと設定が必要です。

```
[Quest 3 / 3S] ~~Wi-Fi~~ [Windows ホスト PC] ==有線 LAN (192.168.123.0/24)== [G1:PC2 (192.168.123.164) / 制御系]
                          teleop_hand_and_arm.py                              teleimager-server(頭のカメラの映像)
                          televuer (https :8012)                              rt/lowstate ⇄ rt/lowcmd / rt/arm_sdk
                          IK (pinocchio + casadi)                             rt/dex3/*
```

---

## 0. 検証の範囲と結果

実機の代わりに Jetson Orin Nano を使い、模擬ロボット(MuJoCo)と画像サーバを動かして検証しました。

| 項目 | 結果 |
|---|---|
| Quest 3S と Windows の接続(televuer、ハンドトラッキング、コントローラー、映像表示) | ✅ 動作 |
| IK(G1_29、pinocchio 3.1.0 + casadi) | ✅ 1 回 4〜10 ms(meshcat を表示しない場合) |
| DDS(Windows と Jetson の間、CycloneDDS 0.10.2) | ✅ 250 通/秒で欠落 0、往復 6.5〜7.6 ms |
| `G1_29_ArmController`(`rt/lowcmd` / `rt/arm_sdk`、CRC、`mode_machine`) | ✅ CRC エラー 0、`mode_machine` 一致 |
| Dex3(dex-retargeting、`rt/dex3/*`) | ✅ 約 50〜73 Hz(パッチ適用後) |
| 画像(Jetson の teleimager → ZMQ → Windows → Quest) | ✅ 動作 |
| 腕の指令の送信レート | ⚠️ 目標 250 Hz に対し約 120〜180 Hz(要調査。「9. 残っている課題」を参照) |

**未検証**:実機が指令を受け付けるかどうか、モード切り替え、WebRTC(PC2 から Quest に映像を直接送る経路)、`--record`、`--sim`、Inspire / BrainCo ハンド。

### 確認したバージョン
| ソフトウェア | バージョン |
|---|---|
| Windows | 11 |
| conda 環境 | Python 3.10、pinocchio 3.1.0(conda-forge)、numpy 1.26.4 |
| xr_teleoperate | main(2026 年 11 月時点)。サブモジュールは televuer `766de45`、teleimager `v2.0.8`、dex-retargeting `d7753d3` |
| unitree_sdk2_python | 1.0.1(cyclonedds 0.10.2 は Windows 用のビルド済みパッケージ) |
| vuer / params-proto | 0.0.60 / 2.13.2 |
| OpenSSL | 3.6.5 |

---

## 1. 前提
- Windows 10 / 11、Miniconda または Anaconda、Git for Windows、OpenSSL
- 有線 LAN の差し込み口(なければ USB 接続の LAN アダプタ)
- Quest と Windows が同じ Wi-Fi(またはテザリング)に接続していること
- コマンドは **Anaconda Prompt(cmd)** で実行する(sshkeyboard のキー入力が確実に届くため)

## 2. 環境構築

```bat
:: conda 環境(README と同じ構成)
conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate tv

:: xr_teleoperate とサブモジュール
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cd xr_teleoperate
git submodule update --init --depth 1

:: teleimager(クライアント側)
cd teleop\teleimager
conda install -c conda-forge libjpeg-turbo
pip install -e .
cd ..\..

:: televuer
cd teleop\televuer
pip install -e .
cd ..\..

:: dex-retargeting(--no-deps を必ず付ける。pip 版の pinocchio "pin" で conda の pinocchio が上書きされるのを防ぐ)
cd teleop\robot_control\dex-retargeting
pip install -e . --no-deps
cd ..\..\..

:: 固定バージョンの依存パッケージ(vuer[all]、params-proto、rerun-sdk、dex-retargeting の依存先 など)
pip install -r <このフォルダ>\requirements-windows.txt

:: unitree_sdk2_python(cyclonedds 0.10.2 の Windows 用 wheel が入る)
cd ..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e . "numpy==1.26.4"
cd ..\xr_teleoperate
```

動作確認:
```bat
python -c "from vuer import Vuer; from televuer import TeleVuer; import pinocchio; from pinocchio import casadi; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; import dex_retargeting; print('ok')"
```

> 注意
> - cmd ではシングルクォートが引用符として扱われないので、`pip install "vuer[all]==0.0.60"` のようにダブルクォートを使う。
> - `pip install rerun` は使わない(名前が同じだけの別のパッケージ)。必要なのは `rerun-sdk==0.20.1`。
> - casadi や `pin` を pip で上から入れない(conda の pinocchio との組み合わせが壊れる)。
> - パッケージを追加したときに numpy が 2.x に上がった場合は、`pip install "numpy==1.26.4"` で戻す。

## 3. パッチの適用(1 回だけ)

`apply_windows_patches.py` を、xr_teleoperate のルートで実行します。2 回実行しても問題ありません。修正前のファイルは `*.orig` として残ります。

```bat
cd xr_teleoperate
python <このフォルダ>\apply_windows_patches.py
```

| 対象ファイル | 修正内容 | 修正しない場合の症状 |
|---|---|---|
| `teleop/televuer/src/televuer/televuer.py` | vuer サーバを子プロセスの中で作るように変更し、画像の共有メモリを子プロセス側でつなぎ直す | `TypeError: self._frozen cannot be converted to a Python object for pickling` |
| 同上 | 共有メモリのサイズを Python の `int` 型で渡す | `TypeError: CreateFileMapping() argument 4 must be int, not numpy.int64` |
| `teleop/teleop_hand_and_arm.py` | タイマーの精度を 1 ms にする(`timeBeginPeriod(1)`) | `time.sleep(4 ms)` が実際には 15.6 ms になり、腕の 250 Hz ループが約 64 Hz に落ちる |
| 同上 | DDS のドメインとインターフェースを環境変数で子プロセスに渡す。`--ipc` を指定したらエラーにする | `--ipc` は Linux 専用の通信方式(`ipc://@...`)を使っているため |
| `teleop/robot_control/robot_hand_unitree.py` | Dex3 の子プロセスが、自分で DDS に接続し、リターゲティングを初期化するように変更 | `TypeError: cannot pickle 'SwigPyObject' object`(`--ee dex3` で起動時に落ちる) |
| (フォルダの作成) | `C:\tmp` を作る | `/tmp/cdds.LOG: cannot open for writing` → `channel factory init error` |

変更は、Linux での動作には影響しません(Windows のときだけ有効になる条件付きの変更です)。

> 参考:`televuer.py` / `televuer_win_spawn.patch` は、televuer だけを先に修正したときの古い方式です。`apply_windows_patches.py` に同じ修正が含まれているので、今は使う必要はありません。置き換え済みの `televuer.py` は、そのままにしておいて問題ありません。

## 4. SSL 証明書(televuer 用)

一部の Windows 版 OpenSSL では、標準の設定ファイルのままだと `v2i_AUTHORITY_KEYID:unknown option ... nonss` というエラーになります。添付の `xr_cert.cnf` の IP アドレス(2 か所)を、**Quest から見た Windows の IP アドレス(Wi-Fi 側)** に書き換えてから実行します。

```bat
cd xr_teleoperate\teleop\televuer
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem -config <このフォルダ>\xr_cert.cnf
mkdir %USERPROFILE%\.config\xr_teleoperate
copy cert.pem %USERPROFILE%\.config\xr_teleoperate\
copy key.pem  %USERPROFILE%\.config\xr_teleoperate\
```

## 5. ネットワーク設定

### 5.1 Windows(PowerShell を管理者として実行)
```powershell
Get-NetAdapter                                                   # 有線 LAN のアダプタ名を確認
Rename-NetAdapter -Name "イーサネット" -NewName "g1eth"          # 英数字の名前に変える(--network-interface で使う)
New-NetIPAddress -InterfaceAlias "g1eth" -IPAddress 192.168.123.2 -PrefixLength 24   # ゲートウェイは設定しない
Set-NetConnectionProfile -InterfaceAlias "g1eth" -NetworkCategory Private              # ケーブル接続後に実行

# ファイアウォール:192.168.123.x からの通信(DDS、ping)と、Quest からの 8012 番ポートへの接続を許可
netsh advfirewall firewall add rule name="G1 net" dir=in action=allow remoteip=192.168.123.0/24 profile=any
netsh advfirewall firewall add rule name="xr_teleop 8012" dir=in action=allow protocol=TCP localport=8012 profile=any
```
- `--network-interface` には **アダプタ名**(この例では `g1eth`)を指定します。CycloneDDS は Windows では表示名(FriendlyName)で照合するので、IP アドレスを指定すると `channel factory init error` になります。
- 有線 LAN にゲートウェイを設定しないことで、インターネットへの接続や Quest との接続は Wi-Fi 側のまま使えます。

### 5.2 確認
```bat
ping 192.168.123.164
python dds_check.py ping --iface g1eth        :: ロボット側で "dds_check.py echo" を動かしておく
```
`received` が約 250/s で、`lost` がほぼ 0 なら OK です。

## 6. ロボット側(G1 の PC2)の画像サーバ

Quest は Wi-Fi で Windows とつながっているので、通常は PC2(192.168.123.164)に直接は届きません。そのため、**映像は WebRTC ではなく ZMQ で Windows を経由して送ります**。PC2 の `~/.config/teleimager/teleimager_server.yaml` の `head_camera` を次のように設定してください。

```yaml
    enable_zmq: true
    zmq_port : 55555
    enable_webrtc: false
```
Quest から PC2 に直接届く構成(ルータを使って Quest も 192.168.123.x にする構成)にすれば、WebRTC も使えます。ただし、今回は検証していません。

Windows 側で、画像が届くかを単体で確認する:
```bat
teleimager-client --host 192.168.123.164
```

## 7. 実行

```bat
conda activate tv
cd xr_teleoperate\teleop
python teleop_hand_and_arm.py --arm G1_29 --ee dex3 --img-server-ip 192.168.123.164 --network-interface g1eth
```

| 引数 | 値 | 説明 |
|---|---|---|
| `--arm` | `G1_29` | G1 の 29 関節版の腕の制御クラス |
| `--ee` | `dex3` など | 手の種類。最初は付けずに腕だけで確認するのがおすすめ |
| `--img-server-ip` | `192.168.123.164` | PC2 の IP アドレス(初期値と同じ) |
| `--network-interface` | `g1eth` | DDS に使う Windows のアダプタ名(Wi-Fi と有線の両方がある場合は必ず指定) |
| `--input-mode` | `hand` / `controller` | ハンドトラッキングかコントローラーか(Dex3 の指を動かせるのは `hand` のときだけ) |
| `--display-mode` | `immersive` / `ego` / `pass-through` | Quest での表示方法 |
| `--motion` | 付ける / 付けない | 付けると `rt/arm_sdk`(モーションモード)、付けないと `rt/lowcmd`(デバッグモード) |
| `--sim` | **付けない** | Isaac Sim 用の設定(DDS のドメインが 1 になる) |
| `--ipc` | **使わない** | Windows では使えない |

1. Quest のブラウザで `https://<Windows の Wi-Fi 側の IP>:8012/?ws=wss://<同じ IP>:8012` を開きます。必ず `https://` から入力してください(省略すると `ERR_EMPTY_RESPONSE` になります)。証明書の警告は「詳細」→「続行」で進みます。
2. 「Virtual Reality」を押します。
3. **teleop を起動したターミナルのウィンドウを選択した状態で** `r` を押すと追従が始まります(`🚀start Tracking🚀` と表示されます)。`q` で終了します。

---

## 8. 実機なしで確認するためのツール(このフォルダ)

| ファイル | 実行する場所 | 用途 |
|---|---|---|
| `test_televuer_local.py` | Windows(`teleop\televuer` の中) | Quest と Windows の間だけの確認(テストパターンの映像を送り、手やコントローラーのデータを表示) |
| `ik_preview.py` | Windows(`teleop` の中) | Quest → IK → MuJoCo で、ロボットの一人称視点(ステレオ表示も可)を Quest に表示。ロボット不要 |
| `dds_check.py` | 両方 | DDS の往復確認(`echo` / `ping`) |
| `g1_mock_robot.py` | Jetson など(Windows でも可) | G1(29 関節)と Dex3 の模擬ロボット。MuJoCo の物理演算付きで、実機と同じ DDS のトピックで通信する |
| `arm_ctrl_check.py` | Windows(`teleop` の中) | xr_teleoperate 本体の `G1_29_ArmController` で模擬ロボットを動かす(XR 機器は不要) |

使用例:
```bat
:: Windows 単体
python test_televuer_local.py --cert cert.pem --key key.pem
python ik_preview.py --cert televuer\cert.pem --key televuer\key.pem --stereo --fpv-size 480x360
python ik_preview.py ... --input controller
```
```bash
# Jetson(模擬ロボット + 画像サーバ)
python g1_mock_robot.py --assets ~/xr_teleoperate/assets/g1 --iface enP8p1s0 --viewer
teleimager-server
```

### 8.1 Jetson(模擬ロボット側)の準備(Ubuntu 22.04、Python 3.10)
```bash
sudo apt install -y git cmake build-essential python3-venv python3-dev v4l-utils
# CycloneDDS 0.10.x(ARM 版 Linux 向けの cyclonedds 0.10.2 のビルド済みパッケージはないため、ソースからビルドする)
cd ~ && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF && cmake --build . --target install -j4
echo 'export CYCLONEDDS_HOME=$HOME/cyclonedds/install' >> ~/.bashrc && source ~/.bashrc
# libjpeg-turbo 3.0 以上(teleimager が使う。apt で入るものは 2.x 系なので、ソースからビルドする)
cd ~ && git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
cd libjpeg-turbo && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/libjpeg-turbo .. && make -j$(nproc) && sudo make install
echo 'export LD_LIBRARY_PATH=/opt/libjpeg-turbo/lib64:/opt/libjpeg-turbo/lib:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
# Python 環境
python3 -m venv ~/xrmock && source ~/xrmock/bin/activate
pip install "numpy==1.26.4" "cyclonedds==0.10.2" mujoco
git clone https://github.com/unitreerobotics/unitree_sdk2_python ~/unitree_sdk2_python && pip install -e ~/unitree_sdk2_python
git clone https://github.com/unitreerobotics/xr_teleoperate ~/xr_teleoperate
git clone -b v2.0.8 https://github.com/unitreerobotics/teleimager ~/teleimager && pip install -e "$HOME/teleimager[server]" "numpy==1.26.4"
# IP アドレス(インターフェース名は ip -br link で確認)
sudo nmcli con add type ethernet ifname enP8p1s0 con-name g1net ipv4.method manual \
     ipv4.addresses 192.168.123.164/24 ipv6.method disabled connection.autoconnect-priority 10
sudo nmcli con up g1net
```
- `/dev/video0` が見つからない(Web カメラが映像デバイスとして認識されない)場合:`sudo modprobe uvcvideo` を試す。それでもだめなら、teleimager の `uvc` ドライバを使う(`pip install -e ".[uvc]"` → `sudo "$(which teleimager-server)" --setup-permissions` → 設定ファイルで `type: uvc` と `vid_pid` を指定)。
- 模擬ロボットのログで `realtime` が 1.0 を下回る場合:`sudo nvpmodel -m 0 && sudo jetson_clocks` で最大性能モードにするか、`--viewer-fps 10` に下げる。

---

## 9. 残っている課題
1. **腕の指令の送信レートが約 120〜180 Hz**(目標 250 Hz)
   - 考えられる原因:同じ Python プロセスの中で、IK や televuer の処理と、腕の指令を送るスレッドが CPU を取り合っている。Windows では CRC の計算が純 Python なので、その分も重くなる。もう一つ、模擬ロボット側の処理が遅れていた影響の可能性もある(修正済み)。
   - 切り分け:①修正版の模擬ロボットで測り直す、②`arm_ctrl_check.py`(IK なし)で測る、③同じ PC を Ubuntu の Live USB で起動して比べる。
2. **WebRTC での映像送信**(PC2 から Quest に直接送る経路)は未検証。
3. **Inspire / BrainCo ハンド**の制御プロセスは、Dex3 と同じ問題がありそうだが、まだパッチを当てていない。
4. **`--motion` でコントローラーを使った歩行**(LocoClient)と、**`--record`** は未検証。
5. 実機が指令を受け付けるか(`mode_machine` の値、デバッグモードへの切り替え)、安全性は、実機でしか確認できない。初めて実機につなぐときは、ロボットを吊るした状態で、すぐ止められる準備をしてから行うこと。

## 10. トラブルシューティング(今回実際に起きたもの)

| 症状 | 原因 | 対処 |
|---|---|---|
| `openssl req` で `v2i_AUTHORITY_KEYID:unknown option ... nonss` | OpenSSL の標準設定ファイルとの相性 | `-config xr_cert.cnf` を指定する |
| `pip install 'vuer[all]'` で `Invalid requirement` | cmd ではシングルクォートが引用符にならない | ダブルクォートを使う |
| `ImportError: cannot import name 'Vuer' from 'vuer'` | `params-proto` 3.x に `Flag` がない(vuer はこの ImportError を「`vuer[all]` を入れてください」という案内で隠してしまう) | `pip install "params-proto==2.13.2"`。本当の原因は `python -c "import vuer.server"` で確認できる |
| `No module named 'teleimager.image_client'` | televuer の例(example)のスクリプトが、古い teleimager 1.x の API を使っている | `test_televuer_local.py` を使う |
| `self._frozen cannot be converted ... for pickling` | televuer が spawn 方式に対応していない | パッチを適用する(3 章) |
| `CreateFileMapping() argument 4 must be int, not numpy.int64` | 共有メモリのサイズに numpy の数値型を渡している | パッチを適用する(3 章) |
| Quest で `ERR_EMPTY_RESPONSE` | `https://` を付けずにアクセスしている | `https://` から入力する |
| `ERR_CONNECTION_TIMED_OUT` | ファイアウォールで止められている | 8012 番ポートを許可し、ネットワークを「プライベート」にする。python.exe のブロック規則がないかも確認する |
| ループが約 21 Hz・腕が約 64 Hz しか出ない | Windows のタイマーの分解能(15.6 ms) | パッチを適用する(`timeBeginPeriod(1)`) |
| IK が 1 回 60〜80 ms かかる | meshcat の表示に時間がかかっている | 表示を無効にする(本体の初期設定では無効) |
| コントローラーを使うと腕が追従しない | 手のデータとコントローラーのデータは、起動時の指定でどちらか一方しか受け取らない | `--input-mode controller` を付けて起動する |
| `/tmp/cdds.LOG: cannot open for writing` → `channel factory init error` | CycloneDDS のログの出力先フォルダがない | `mkdir C:\tmp`(パッチスクリプトでも作成される) |
| `--network-interface <IP アドレス>` で `channel factory init error` | IP アドレスは使えない | アダプタ名を指定する |
| `No module named 'rerun'` / `rerun.blueprint` | 必要なパッケージは `rerun-sdk` | `pip uninstall rerun` → `pip install rerun-sdk==0.20.1` |
| `cannot pickle 'SwigPyObject'`(`--ee dex3`) | Dex3 の子プロセスが spawn 方式に対応していない | パッチを適用する(3 章) |
| `Image server is unreachable` / `Failed to get camera configuration` | PC2 で teleimager が動いていない、またはポート 60000 に届かない | `teleimager-server` を起動し、`teleimager-client --host ...` で確認する |
| `Enter debug mode: Failed` / `[ClientStub] send request error` | モード切り替えの要求に応答がない(模擬ロボットでは毎回出る) | 模擬ロボットの場合は無視してよい。実機の場合は、ロボットの状態を確認する |
| `r` を押しても追従が始まらない | キー入力がターミナルに届いていない | ターミナルを選択してから押す。Anaconda Prompt か cmd を使う |
| (Jetson)`Cannot find V4L2Camera ... video_id 0` | Web カメラが映像デバイスとして認識されていない | 8.1 の注記を参照 |
| (Jetson)模擬ロボットが `realtime x0.3` | ビューアーの描画が物理演算を止めていた | 修正済みの `g1_mock_robot.py` を使う |
