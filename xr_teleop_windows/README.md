# xr_teleoperate を Windows で使うための修正とテスト

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
