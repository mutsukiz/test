# xr_teleoperate を Windows で使うための修正とテスト

**手順書は [`WINDOWS_HOST_GUIDE.md`](WINDOWS_HOST_GUIDE.md) を参照してください。** パッチは `apply_windows_patches.py` 1 本にまとめてあり(televuer の修正も含む)、依存パッケージは `requirements-windows.txt`、証明書の設定ファイルは `xr_cert.cnf` です。

## televuer の Windows 対応(spawn 対応)

Windows の multiprocessing は `spawn` 方式です。そのため元の televuer は、`Process(target=self._vuer_run)` で TeleVuer オブジェクトを pickle しようとして、次のエラーで失敗します。

```
TypeError: self._frozen cannot be converted to a Python object for pickling
```

この修正で、vuer サーバの作成を子プロセス側に移しました。また、画像の共有メモリを子プロセス側でもう一度つなぎ直すようにしました。Linux の `fork` 方式でも、これまでどおり動きます。

- 対象:televuer のコミット `766de45`(xr_teleoperate のサブモジュールが固定しているもの)
- `televuer.py`:修正済みのファイル。`teleop/televuer/src/televuer/televuer.py` と置き換えます。
- `televuer_win_spawn.patch`:同じ修正の差分ファイル(`git apply` 用)

## test_televuer_local.py

Quest と Windows の間だけを確認するテストです。teleimager もロボットも要りません。テストパターンの画像を Quest に送り、手やコントローラのデータが届いているかを表示します。

## 実機なしで確認するためのツール

| ファイル | 置き場所 | 内容 |
|---|---|---|
| `ik_preview.py` | `xr_teleoperate/teleop/` | Quest → IK → MuJoCo による一人称視点(ロボット不要、Windows 単体で動く) |
| `g1_mock_robot.py` | どこでも可(`--assets` で G1 のモデルを指定) | G1(29 関節)と Dex3 の模擬ロボット。MuJoCo で物理演算し、`rt/lowcmd` / `rt/arm_sdk` / `rt/dex3/*` を受け取り、`rt/lowstate` / `rt/dex3/*/state` を返す |
| `arm_ctrl_check.py` | `xr_teleoperate/teleop/` | xr_teleoperate 本体の `G1_29_ArmController` で模擬ロボットを動かす(XR 機器は不要) |
| `dds_check.py` | どこでも可 | 2 台の間の DDS の往復確認(ping / echo)。届いた割合、欠落、往復時間を表示する |
| `apply_windows_patches.py` | `xr_teleoperate/` のルートで実行 | `teleop_hand_and_arm.py` を Windows で動かすための修正。タイマーを 1 ms にする、Dex3 の制御プロセスを spawn 方式に対応させる、`--ipc` を使えないようにする |

### Windows で teleop_hand_and_arm.py を使うときの注意
- `mkdir C:\tmp` を実行しておく(unitree_sdk2_python は `--network-interface` を指定すると、CycloneDDS のログを `/tmp/cdds.LOG` に書き出す設定になっており、フォルダがないと DDS の初期化に失敗する)
- `--network-interface` には、Windows のアダプタ名(コントロールパネルに表示される名前)を指定する。IP アドレスは使えない
- `pip install -r requirements.txt` を実行しておく(`rerun` ではなく `rerun-sdk==0.20.1` が必要)
- `--ipc` は使えない(Linux 専用の通信方式を使っているため)
