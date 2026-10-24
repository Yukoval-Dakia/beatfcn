# 相位探针（phase probe）— 2026-10-23

设计依据：design/FIRST_CANDIDATE_v2.md section 5 row 1, section 3.3 time head v2。问题：40 ms 格内的起音偏移（0–39 ms）能否从冻结 PupuM2D 格 token 读出，与 10 ms mel 相比如何。

## 结论

**预注册判据**：remove the mel branch if MAE_A_all - MAE_B <= 1 ms and MAE_A_all - MAE_C <= 1 ms (test split, expectation decoder, models selected on val MAE)。MAE_A_all − MAE_B = +0.56 ms，MAE_A_all − MAE_C = +0.66 ms → **满足"删 mel"判据**。按模式：mania +0.51 / +0.82 ms，phira +0.62 / +0.46 ms。
5 折（按歌）稳健性：A_all − B = +1.34 ± 0.67 ms，A_all − C = +1.60 ± 0.83 ms（折间 sd）。
**严格检查**（val 上最优的 token 表示 A_all_rpd 对最优的 mel 表示 B_heur，含非学习的谱通量启发式 B_heur）：全体 +0.24 ms，mania +1.52 ms，phira -1.43 ms；对 C_flux：全体 -0.04，mania -0.09 ms → **不通过**（阈值 1 ms）。
5 折：A_all_rpd − B_heur = -1.43 ± 1.18 ms（mania -1.22 ± 2.14）。
非学习启发式 B_heur（mel 谱通量选 10 ms 桶）MAE = 7.52 ms（mania 5.33）；D（仅边缘分布）MAE = 10.07 ms；E 比 A_all 的 MAE 低 +0.32 ms（节律先验）。

### 解读

- 条件 1（预注册判据，固定切分）：满足（A_all−B +0.56，A_all−C +0.66 ms）。
- 条件 2（同判据，5 折均值）：不满足（+1.34 ± 0.67，+1.60 ± 0.83 ms）；预注册的 A_all（两级 PCA 到 2048 维）是最差的 token 表示（5 折 9.25 ms，对比随机投影 A_all_rp 7.30、A_all_rpd 7.18 ms），它在固定切分上的“满足”是弱对弱的巧合。
- 条件 3（最优 token 表示 A_all_rpd 对最优学习型 mel 表示 B_flux 与对 token+mel 的 C_flux，5 折均值）：满足（-0.42 ± 0.28，-0.02 ± 0.03 ms）：学习型读出下 token 不劣于 mel，且 mel 叠加到 token 上没有增益。
- 条件 4（A_all_rpd 对非学习的 mel 谱通量启发式 B_heur，mania，固定切分）：不满足（+1.52 ms；5 折 -1.22 ± 2.14 ms，折间不稳定）：mel 里仍有一部分相位信息是零参数启发式能读出、而 38 首歌训练的 token 探针在该切分上读不出的；phira 上启发式无信息（10.38 ms ≈ 机会水平）。

**按预注册判据：删 mel 支路。稳健性支持“token 不劣于学习型 mel 读出、mel 无叠加增益”，但未能证明 token 已含 mel 的全部相位信息（条件 4 不满足）。建议：删 mel 支路，但保留 §5 实验 4 的 mel × 间隔核 2×2 对照作为验证；所有音频探针都远不及节律先验（E_only）。**

## 主表（测试集 = 预注册的按歌 80/20 切分；模型按 val MAE 选网格配置）

| 模型 | 输入维 | 解码 | MAE ms | 中位 ms | P(≤5) | P(≤10) | NLL | top1 | MAE mania | MAE phira |
|---|---|---|---|---|---|---|---|---|---|---|
| D | - | argmax | 16.89 | 16.0 | 0.221 | 0.349 | 3.693 | 0.019 | 17.41 | 16.23 |
| D | - | expect | 10.07 | 10.2 | 0.248 | 0.487 | 3.693 | 0.019 | 10.16 | 9.95 |
| D_mode | - | argmax | 10.72 | 9.0 | 0.283 | 0.540 | 3.694 | 0.020 | 10.37 | 11.18 |
| D_mode | - | expect | 10.06 | 9.8 | 0.249 | 0.502 | 3.694 | 0.020 | 10.17 | 9.92 |
| A_9 | 2040 | argmax | 9.18 | 6.0 | 0.447 | 0.685 | 3.727 | 0.050 | 7.98 | 10.75 |
| A_9 | 2040 | expect | 7.72 | 6.2 | 0.404 | 0.728 | 3.727 | 0.050 | 6.96 | 8.71 |
| A_16 | 2040 | argmax | 9.26 | 7.0 | 0.433 | 0.677 | 3.798 | 0.050 | 7.98 | 10.93 |
| A_16 | 2040 | expect | 7.85 | 6.4 | 0.389 | 0.734 | 3.798 | 0.050 | 7.05 | 8.88 |
| A_24 | 2040 | argmax | 9.77 | 7.0 | 0.415 | 0.650 | 3.760 | 0.054 | 8.44 | 11.50 |
| A_24 | 2040 | expect | 8.11 | 6.7 | 0.373 | 0.702 | 3.760 | 0.054 | 7.22 | 9.27 |
| A_all | 2048 | argmax | 9.77 | 7.0 | 0.405 | 0.641 | 3.666 | 0.052 | 8.34 | 11.62 |
| A_all | 2048 | expect | 8.76 | 8.3 | 0.301 | 0.600 | 3.666 | 0.052 | 8.21 | 9.48 |
| A_all_rp | 2046 | argmax | 9.27 | 7.0 | 0.435 | 0.672 | 3.731 | 0.057 | 7.84 | 11.14 |
| A_all_rp | 2046 | expect | 7.99 | 6.6 | 0.376 | 0.711 | 3.731 | 0.057 | 7.26 | 8.95 |
| A_all_rpd | 4092 | argmax | 8.95 | 6.0 | 0.452 | 0.696 | 3.812 | 0.056 | 7.38 | 11.00 |
| A_all_rpd | 4092 | expect | 7.76 | 6.2 | 0.408 | 0.734 | 3.812 | 0.056 | 6.85 | 8.95 |
| B | 1024 | argmax | 9.03 | 6.0 | 0.440 | 0.680 | 3.649 | 0.052 | 7.77 | 10.66 |
| B | 1024 | expect | 8.20 | 7.1 | 0.357 | 0.664 | 3.649 | 0.052 | 7.70 | 8.86 |
| B_flux | 1927 | argmax | 8.90 | 6.0 | 0.440 | 0.689 | 3.704 | 0.047 | 7.65 | 10.51 |
| B_flux | 1927 | expect | 7.85 | 6.6 | 0.387 | 0.701 | 3.704 | 0.047 | 7.24 | 8.66 |
| B_heur | - | argmax | 8.55 | 6.0 | 0.420 | 0.706 | 10.860 | 0.049 | 6.19 | 11.62 |
| B_heur | - | expect | 7.52 | 4.5 | 0.535 | 0.744 | 10.860 | 0.049 | 5.33 | 10.38 |
| C | 3072 | argmax | 9.27 | 7.0 | 0.429 | 0.669 | 3.708 | 0.050 | 8.06 | 10.86 |
| C | 3072 | expect | 8.10 | 6.9 | 0.357 | 0.687 | 3.708 | 0.050 | 7.40 | 9.01 |
| C_flux | 6019 | argmax | 9.04 | 7.0 | 0.438 | 0.689 | 3.849 | 0.053 | 7.65 | 10.86 |
| C_flux | 6019 | expect | 7.80 | 6.3 | 0.403 | 0.728 | 3.849 | 0.053 | 6.94 | 8.93 |
| E | 2088 | argmax | 9.50 | 7.0 | 0.424 | 0.655 | 3.616 | 0.060 | 8.05 | 11.39 |
| E | 2088 | expect | 8.44 | 7.8 | 0.317 | 0.631 | 3.616 | 0.060 | 7.91 | 9.13 |
| E_only | 40 | argmax | 4.77 | 1.0 | 0.792 | 0.849 | 2.859 | 0.217 | 3.44 | 6.50 |
| E_only | 40 | expect | 5.24 | 2.4 | 0.645 | 0.810 | 2.859 | 0.217 | 4.49 | 6.21 |

## 5 折（按歌）MAE，expect 解码，均值 ± 折间 sd

| 模型 | 全体 | mania | phira | 各折（全体） |
|---|---|---|---|---|
| A_all | 9.25 ± 0.28 | 9.07 ± 0.50 | 9.46 ± 0.05 | 8.76, 9.30, 9.38, 9.36, 9.47 |
| A_all_rp | 7.30 ± 0.85 | 6.35 ± 0.84 | 8.12 ± 1.08 | 7.99, 6.94, 6.37, 6.82, 8.38 |
| A_all_rpd | 7.18 ± 0.90 | 6.12 ± 0.76 | 8.08 ± 1.28 | 7.76, 6.91, 6.15, 6.67, 8.42 |
| B | 7.92 ± 0.59 | 7.09 ± 0.65 | 8.57 ± 0.83 | 8.20, 7.48, 7.41, 7.69, 8.81 |
| B_flux | 7.60 ± 0.73 | 6.73 ± 0.64 | 8.30 ± 1.06 | 7.85, 7.19, 6.97, 7.22, 8.77 |
| C | 7.65 ± 0.73 | 6.76 ± 0.73 | 8.38 ± 1.02 | 8.10, 7.18, 6.88, 7.40, 8.70 |
| C_flux | 7.20 ± 0.91 | 6.15 ± 0.77 | 8.09 ± 1.29 | 7.80, 6.88, 6.17, 6.70, 8.45 |
| E | 9.11 ± 0.38 | 8.91 ± 0.57 | 9.32 ± 0.13 | 8.44, 9.20, 9.33, 9.25, 9.31 |
| E_only | 5.25 ± 0.53 | 4.95 ± 0.58 | 5.59 ± 0.66 | 5.24, 5.87, 4.95, 4.54, 5.62 |
| D | 9.96 ± 0.14 | 10.01 ± 0.11 | 9.93 ± 0.17 | 10.07, 9.96, 10.10, 9.91, 9.76 |
| B_heur | 8.62 ± 1.42 | 7.34 ± 1.83 | 10.18 ± 1.23 | 7.52, 8.31, 8.67, 7.58, 10.99 |

差值（expect MAE，折均值 ± sd；括号内 mania）：A_all-B = +1.34 ± 0.67（+1.97 ± 0.93）；A_all-C = +1.60 ± 0.83（+2.31 ± 0.98）；A_all_rpd-B_flux = -0.42 ± 0.28（-0.60 ± 0.25）；A_all_rpd-B_heur = -1.43 ± 1.18（-1.22 ± 2.14）；A_all_rpd-C_flux = -0.02 ± 0.03（-0.03 ± 0.06）

## 按模式：mania（测试样本 7331）

| 模型 | 解码 | MAE ms | 中位 ms | P(≤5) | P(≤10) |
|---|---|---|---|---|---|
| D | argmax | 17.41 | 17.0 | 0.219 | 0.335 |
| D | expect | 10.16 | 10.2 | 0.246 | 0.491 |
| D_mode | argmax | 10.37 | 10.0 | 0.270 | 0.510 |
| D_mode | expect | 10.17 | 10.5 | 0.246 | 0.491 |
| A_9 | argmax | 7.98 | 6.0 | 0.461 | 0.727 |
| A_9 | expect | 6.96 | 6.3 | 0.386 | 0.784 |
| A_16 | argmax | 7.98 | 6.0 | 0.443 | 0.719 |
| A_16 | expect | 7.05 | 6.4 | 0.366 | 0.795 |
| A_24 | argmax | 8.44 | 7.0 | 0.431 | 0.687 |
| A_24 | expect | 7.22 | 6.3 | 0.377 | 0.751 |
| A_all | argmax | 8.34 | 6.0 | 0.440 | 0.691 |
| A_all | expect | 8.21 | 8.0 | 0.316 | 0.627 |
| A_all_rp | argmax | 7.84 | 6.0 | 0.452 | 0.718 |
| A_all_rp | expect | 7.26 | 6.7 | 0.352 | 0.757 |
| A_all_rpd | argmax | 7.38 | 6.0 | 0.472 | 0.753 |
| A_all_rpd | expect | 6.85 | 6.2 | 0.393 | 0.791 |
| B | argmax | 7.77 | 6.0 | 0.462 | 0.732 |
| B | expect | 7.70 | 7.0 | 0.355 | 0.691 |
| B_flux | argmax | 7.65 | 6.0 | 0.456 | 0.750 |
| B_flux | expect | 7.24 | 6.4 | 0.380 | 0.739 |
| B_heur | argmax | 6.19 | 5.0 | 0.572 | 0.855 |
| B_heur | expect | 5.33 | 3.5 | 0.679 | 0.878 |
| C | argmax | 8.06 | 6.0 | 0.440 | 0.710 |
| C | expect | 7.40 | 6.8 | 0.341 | 0.732 |
| C_flux | argmax | 7.65 | 6.0 | 0.449 | 0.740 |
| C_flux | expect | 6.94 | 6.3 | 0.380 | 0.783 |
| E | argmax | 8.05 | 6.0 | 0.458 | 0.710 |
| E | expect | 7.91 | 7.6 | 0.329 | 0.659 |
| E_only | argmax | 3.44 | 1.0 | 0.857 | 0.903 |
| E_only | expect | 4.49 | 1.7 | 0.690 | 0.845 |

## 按模式：phira（测试样本 5625）

| 模型 | 解码 | MAE ms | 中位 ms | P(≤5) | P(≤10) |
|---|---|---|---|---|---|
| D | argmax | 16.23 | 15.0 | 0.224 | 0.368 |
| D | expect | 9.95 | 10.2 | 0.252 | 0.482 |
| D_mode | argmax | 11.18 | 9.0 | 0.302 | 0.579 |
| D_mode | expect | 9.92 | 9.8 | 0.254 | 0.516 |
| A_9 | argmax | 10.75 | 7.0 | 0.430 | 0.630 |
| A_9 | expect | 8.71 | 6.1 | 0.427 | 0.656 |
| A_16 | argmax | 10.93 | 8.0 | 0.421 | 0.623 |
| A_16 | expect | 8.88 | 6.4 | 0.420 | 0.653 |
| A_24 | argmax | 11.50 | 8.0 | 0.394 | 0.603 |
| A_24 | expect | 9.27 | 7.2 | 0.368 | 0.638 |
| A_all | argmax | 11.62 | 9.0 | 0.359 | 0.575 |
| A_all | expect | 9.48 | 8.7 | 0.282 | 0.566 |
| A_all_rp | argmax | 11.14 | 8.0 | 0.414 | 0.612 |
| A_all_rp | expect | 8.95 | 6.5 | 0.407 | 0.651 |
| A_all_rpd | argmax | 11.00 | 8.0 | 0.427 | 0.622 |
| A_all_rpd | expect | 8.95 | 6.2 | 0.428 | 0.659 |
| B | argmax | 10.66 | 8.0 | 0.412 | 0.613 |
| B | expect | 8.86 | 7.4 | 0.361 | 0.628 |
| B_flux | argmax | 10.51 | 8.0 | 0.420 | 0.609 |
| B_flux | expect | 8.66 | 6.8 | 0.395 | 0.652 |
| B_heur | argmax | 11.62 | 10.0 | 0.222 | 0.511 |
| B_heur | expect | 10.38 | 8.5 | 0.348 | 0.568 |
| C | argmax | 10.86 | 8.0 | 0.414 | 0.615 |
| C | expect | 9.01 | 7.1 | 0.378 | 0.629 |
| C_flux | argmax | 10.86 | 8.0 | 0.425 | 0.621 |
| C_flux | expect | 8.93 | 6.1 | 0.433 | 0.657 |
| E | argmax | 11.39 | 8.0 | 0.379 | 0.585 |
| E | expect | 9.13 | 8.2 | 0.302 | 0.595 |
| E_only | argmax | 6.50 | 2.0 | 0.707 | 0.780 |
| E_only | expect | 6.21 | 3.4 | 0.587 | 0.765 |

## 数据

- 歌曲：60（{'osu': 30, 'phira': 30}），train/val/test 歌曲数 38/10/12；使用谱面 61 张（这些歌在清单中共 177 张，只用本地已有 canon 的）。
- 样本（含起音的 40 ms 格，按 (chart, cell) 去重）：共 67955，train 45883 / val 9116 / test 12956；测试集按模式 {'mania': 7331, 'phira': 5625}。
- 下载（`modal volume get`，只读）：taps 5.870 GB（180 个文件，60 首 × L9/L16/L24；本次运行新下载 0.000 GB，其余来自前一步 `--download-only`）+ 清单/PCA 34.4 MB = 共 5.905 GB，全部仍在 /private/tmp/claude-501/-Users-yukoval-yukoval-projects-beatfcn/247c416f-2bc5-4a80-b99c-5dfe8bb8f8b0/scratchpad/probe（未删除；特征缓存 cache/ 为派生数据）。
- 清单：charts.offset-v2-3d99ff6bd5e7.jsonl（split=train，offset decision v2，训练同款筛选）；taps：v4_i8_L9-16-24；pca：i8_9-16-24。

## 时间映射（与训练一致）

- 模型视角起音：`native onset_us + (offset_decision.shift_ms + offset_decision.c_ms) * 1000`；目标时刻 = `quantize_us(model-view onset_us) = ((us + 500) // 1000) * 1000 (hidden targets are quantised to 1 ms, round half up)`；格 = `quantised onset_us // 40_000`；格内 ms = `(quantised onset_us % 40_000) // 1000 (exact integer after quantisation)`。
- 音频轴：tap cell k <-> taps[k]; mel10 frames 4k..4k+3; neighbours out of range -> zero tokens。
- 反量化：int8 * scale_l from pca/i8_9-16-24.npz (whiten=False), then per-token LayerNorm (no affine, eps 1e-5)。
- canon events are playable only (fakes are kept separately), as chart_from_canon reads doc["events"]。
- 出处：
  - beatfcn/train/data.py:72-75 model_shift_us (shift_ms + c_ms when apply_codec_constant)
  - beatfcn/train/data.py:127 Corpus.__init__ apply_codec_constant=True (default, used by train_app)
  - beatfcn/train/data.py:157-162 Corpus._load_chart -> chart_from_canon(doc, model_shift_us(r, self.apply_c))
  - beatfcn/train/data.py:98-107 chart_from_canon adds shift_us to onset_us / end_us
  - beatfcn/events/canonical.py:286-287 quantize_us; :333-345 build_model_view quantises hidden targets (not observed anchors)
  - beatfcn/events/requests.py:281 sample_request -> build_model_view; train/loop.py:81-87 sample_request on corpus.chart, then choose_core/build_core
  - beatfcn/model/features.py:22 CELL_US = 40_000; :23 BIN_US = 10_000
  - beatfcn/model/features.py:306-320 choose_core: c0 is a multiple of CELL_US
  - beatfcn/model/features.py:113-125 target_group: t = time_us - c0, bin = t // BIN_US, residual = t % BIN_US // MS
  - beatfcn/model/batch.py:221 collate_meta: cell label at t_rel_us // CELL_US
  - beatfcn/model/batch.py:316-317 attach_audio: taps sliced at song_cell0, mel at song_cell0 * FRAMES_PER_CELL
  - beatfcn/train/data.py:111-123 QuantArray: int8 * scale_l (pca npz), mean not re-added; batch.py:371-384 dequant same
  - beatfcn/model/p1.py:222,504 tap_ln = nn.LayerNorm(tap_dim) per tap applied to each token

## 对齐自检（谱面→音频映射）

量化后的模型视角起音周围的谱通量包络（`ingest/offset.flux_envelope`，mel10），帧偏移 −6..+6 的均值，以及偏移 0 处按帧内 ms 分组的均值（每源前 8 首）：

- osu（12421 起音）：按帧偏移 [0.014, -0.082, -0.113, -0.223, -0.318, -0.058, 0.628, 0.989, 0.445, -0.27, -0.468, -0.336, -0.085]；偏移 0 按帧内 ms [0.849, 0.982, 0.669, 0.734, 0.591, 0.561, 0.823, 0.439, 0.247, 0.143]
- phira（9843 起音）：按帧偏移 [0.033, -0.093, -0.184, -0.163, -0.027, 0.294, 0.568, 0.457, 0.184, 0.037, -0.058, -0.152, -0.146]；偏移 0 按帧内 ms [0.655, 0.602, 0.695, 0.53, 0.624, 0.755, 0.434, 0.574, 0.31, 0.605]

## 特征与模型

- **A_tap**：cells c-1,c,c+1 x 8 bands x 1024 (dequant + per-token LN) -> per-token PCA 85 dims fitted on train+val songs -> 2040 dims; variance kept per layer: L9 0.660, L16 0.661, L24 0.493
- **A_all**：concat of the three A_tap features (6120) -> second-stage PCA 2048 dims (train+val rows), variance kept 0.924
- **A_all_rp**：reduction check: fixed Gaussian random projection (seed 12345) of each tap's 3 x 8 x 1024 block to 682 dims, 3 taps concatenated = 2046 dims (no fitting)
- **A_all_rpd**：A_all_rp + random projection (seed 54321, 682 dims per tap) of relu(token(c) - token(c-1)), relu(token(c+1) - token(c)) = 4092 dims (token analogue of spectral flux)
- **B**：mel10 frames 4c-2 .. 4c+5 (8 x 128 = 1024 dims, raw Pupu-normalised log-mel)
- **B_flux**：B + rectified frame differences (7 x 128) + per-frame summed flux (7) = 1927 dims
- **B_heur**：non-learned: 10 ms bin b = argmax over b of summed positive flux at frame 4c+b+1, prediction = bin centre 10b+4.5 (expect decoder; argmax gives 10b)
- **C_flux**：A_all_rpd + B_flux
- **C**：A_all + B
- **D**：train marginal of the 40 offsets (chance); D_mode = per-mode marginal
- **E**：A_all + rhythm context (40 dims): gaps from the cell start back to the previous 4 distinct onsets of the same chart and the 4 IOIs among the previous 5; each as seconds, log1p(ms), sin/cos of the 40 ms phase, + presence flags
- **E_only**：rhythm context alone (diagnostic)
- **standardisation**：every input z-scored per dim on train+val rows
- **model**：Dropout(in_drop) - Linear(d, 256) - GELU - Dropout(0.3) - Linear(256, 40); AdamW wd 0.1, batch 512, cosine over 80 epochs, early stop on val NLL (patience 8), CPU. Per feature set the same grid lr x in_drop = {1e-4,1e-3} x {0,0.5} is run; the model with the lowest val MAE (expectation decoder) is reported (results), the lowest-val-NLL one is kept in results_nll_selected. val = held-out songs. (Run 1 with a single lr 1e-3 / wd 1e-2 setting over-fitted within one epoch on every audio set; run 2 with a wd axis showed wd has no effect.)
- **cv**：5 by-song folds from one seeded shuffle (fold 0 = the reported split); folds 1-4 reuse each set's fold-0 configuration; PCA / standardisation statistics stay those of fold 0 (unsupervised, mild leakage into folds 1-4)
- **decoders**：argmax class; expectation sum_k p_k k. Errors against the integer target (0-39 ms); mae_vs_exact_ms uses the sub-ms onset position

## 训练细节

| 模型 | 输入维 | 参数 | 选中 lr/in_drop | 轮数 | val NLL | val MAE(expect) | 秒（整个网格） |
|---|---|---|---|---|---|---|---|
| A_9 | 2040 | 532776 | 1e-03/0.5 | 9 | 3.829 | 7.25 | 20.1 |
| A_16 | 2040 | 532776 | 1e-03/0.5 | 9 | 3.842 | 7.46 | 15.6 |
| A_24 | 2040 | 532776 | 1e-03/0.5 | 9 | 3.810 | 7.44 | 15.8 |
| A_all | 2048 | 534824 | 1e-03/0.5 | 10 | 3.653 | 8.58 | 23.9 |
| A_all_rp | 2046 | 534312 | 1e-03/0.5 | 9 | 3.751 | 7.27 | 15.3 |
| A_all_rpd | 4092 | 1058088 | 1e-03/0.5 | 9 | 3.792 | 7.16 | 29.9 |
| B | 1024 | 272680 | 1e-03/0.0 | 9 | 3.553 | 7.68 | 13.8 |
| B_flux | 1927 | 503848 | 1e-03/0.0 | 9 | 3.601 | 7.43 | 16.9 |
| C | 3072 | 796968 | 1e-03/0.0 | 9 | 3.563 | 7.41 | 27.5 |
| C_flux | 6019 | 1551400 | 1e-03/0.5 | 9 | 3.807 | 7.17 | 55.9 |
| E | 2088 | 545064 | 1e-03/0.5 | 10 | 3.636 | 8.35 | 31.3 |
| E_only | 40 | 20776 | 1e-03/0.0 | 22 | 3.144 | 5.88 | 28.6 |

按 val NLL 选配置时的测试 MAE（expect）：A_9 8.46，A_16 8.63，A_24 8.62，A_all 8.83，A_all_rp 8.46，A_all_rpd 8.03，B 8.41，B_flux 8.30，C 8.28，C_flux 8.14，E 8.51，E_only 5.24

训练集偏移边缘分布（0..39 ms）：[0.026, 0.023, 0.03, 0.021, 0.02, 0.027, 0.03, 0.028, 0.026, 0.021, 0.029, 0.027, 0.026, 0.025, 0.022, 0.027, 0.029, 0.027, 0.025, 0.025, 0.023, 0.023, 0.029, 0.022, 0.021, 0.028, 0.028, 0.029, 0.024, 0.018, 0.026, 0.026, 0.024, 0.025, 0.02, 0.021, 0.029, 0.027, 0.022, 0.02]

运行时长 656.8 s。脚本：`scripts/phase_probe.py --root <root> [--download]`。
