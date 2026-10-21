# アクリル球レンズ (Φ80 + Edmund #48-233 + #47-917)

f≈65mm / F5.6 / APS-C(像高14.2mmまで)。絞りは球の前0.3mmに固定、球は切らない。

| 面 | 部品 | R (mm) | 厚さ/間隔 (mm) |
| --- | --- | --- | --- |
| 絞り | 開口Φ11.6 (F5.6) | | 0.3 |
| 1-2 | アクリル球 Φ80 (nd1.4917, νd57.4) | 40 / -40 | 80.0 → 0.5 |
| 3-4 | #48-233 平凸 N-SF5 Φ30 (凸面を球側) | 33.635 / ∞ | 5.78 → 1.74 |
| 5-6 | #47-917 両凹 N-SF11 Φ25 | -39.884 / 39.884 | 3.00 → センサーまで11.38 |

- `BallLens_F*_lenssim.txt` : Lens Sim 用 (絞りがレンズ前のため `aperture idx = 0` とした。Lens Sim上の動作は未確認)
- `*_layout.png` 光線図 / `*_spots.png` スポット / `*_simulation.png` 写りのシミュレーション / `*_prescription.csv`
- 中心厚は #48-233, #47-917 ともカタログからの推定値 (実寸未確認)。
- 光線追跡は添付 lens_handover の lenslab を使用 (`ballsys.py` でアクリル球をカタログに追加)。再現: `python build_outputs.py` (lens_handover の場所は ballsys.py の ROOT を直す)
