# 训练框架 review 指南（2026-10-23 版）

目标读者：要审阅 P1 训练代码的人。按数据流顺序读，每节给出文件、入口函数、它保证的不变量，以及"审阅时值得盯的点"。行号对应提交 `f4d0da4`。

## 0. 一张图

```
manifest/charts.jsonl ─┐
canon/<src>/<id>.json ─┤  Corpus (train/data.py)        每个 DataLoader worker 一份
mel10/<sha>.npy ───────┘       │
                               ▼
              CoreSampler.draw  (train/loop.py:110)      歌曲常驻池 → 谱面 → 请求 → 片段(CoreExample)
                               │
                               ▼
              collate_meta      (model/batch.py:87)      元数据张量 + 音频配方 `_audio`，约 2.9 MB/批
                               │  进程间传输（DataLoader）
                               ▼
主进程：SongPool.get (train/pool.py:73)  ──►  attach_audio (model/batch.py:219)   显存里切出 audio / mel / summary / ref_audio / hold_*
                               │
                               ▼
              GraphedStep.fill → step (train/graph.py)   或  to_device + P1Model.forward（非图路径）
                               │
                               ▼
              P1Model.forward (model/p1.py:626)          encode → time_logits → residual → _inner(GRU) → _hold
                               │
                               ▼
              merge / summarize (train/loop.py:160/173)  指标留在设备上，只在打日志时同步
```

云端编排在 `beatfcn/cloud/train_app.py`：`prep_train`（CPU 预处理）、`train`（主循环）、`sweep`（多任务）、各种 bench / diag。

## 1. 语料与采样（`beatfcn/train/data.py`, `beatfcn/train/loop.py`）

- `load_manifest`（data.py:30）与 `selected_for_training`（:58）：训练用哪些谱面。规则来自 `design/DATA_RULES.md`。
- `Corpus`（:122）：
  - `chart(chart_id)`（:144）：读 canon JSON → `Chart`，应用来源常数偏移 `model_shift_us`。缓存 512 条，**超限整体清空**（不是 LRU；`salient` 缓存同理，256 条）。
  - `song_info(sha)`（:189）：只读 `.npy` 文件头得到格数、摘要行数（`npy_shape`，:86，不做 mmap——网络盘上 mmap 和目录扫描都极慢，见 §7）。
  - `load_numpy(sha)`（:211）：整首歌的 int8 tap、S20 摘要、mel，供主进程的歌曲池。
  - `song(sha, in_memory)`（:221）：旧的 CPU 路径（验证、生成、测试仍在用）。
- `CoreSampler`（loop.py:37）：
  - `_resident`（:89）：每个 worker 按模式常驻 `pool` 首歌，每首用 `residence` 次后换；池子**逐次增长**（第一批只读一首歌）。
  - `core`（:62）：请求采样 → `build_core` → 先查 `MAX_NOTES` 和静态上限 `limits`，再查目标合法性 `check_targets`。所有拒绝都是 `SkipExample`，由 `draw` 计数到 `skipped`，日志里能看到。
  - `draw`（:110）：一次给同一首歌抽最多 4 份；100 次失败才报错。
  - 审阅点：采样分布。模式权重 ∝ √该模式的歌曲数；歌曲在池内均匀；常驻复用让同一首歌的样本在时间上相邻（配对训练两套模型吃的是同一批，所以对比是公平的）。
  - `core` 的合法性：长条终点上限只来自可编辑区间和**已知**事件（observed + 已生成前缀），不含隐藏的未来目标；开始于 4 s 光环之前但仍在持续的长条算作已生成上下文（`tests/test_model_features.py::HoldContextTests`）。Phira 不超过 4 个的叠键按 DATA_RULES v4 放行（`tests/test_events_canonical.py::SourceStackTests`）。

## 2. 打包（`beatfcn/model/batch.py`）

- `collate_meta`（:87）：所有不需要音频的张量，外加 `_audio` 配方（歌曲 key、起始格、参考簇读取范围、长条块位置、摘要池化系数）。
  - 维度要么按 `BUCKET` 向上取整，要么在 `static=STATIC` 时固定（CUDA graph 需要）；超限抛 `SkipExample`。
  - `hold_*` 按 `H` 补齐，`hold_valid` 标记真伪。
- `attach_audio`（:219）：按配方从常驻歌曲切片，任意设备。`_dslice` 与旧 `_slice` 语义一致（越界补零）。
- `collate`（:277）= `collate_meta` + CPU 上的 `attach_audio`。**不变量的边界**：共用函数只保证两条路径的计算相同，不保证流安全和 static/dynamic 形状语义相同；`tests/test_transport.py::PoolTests` 逐张量比对 CPU 上的结果。静态模式下摘要上限 `S=256`，超过 8.5 分钟的歌比非静态路径（上限 600）池化更粗——这是模型输入的语义变化，评估图化收益时要与它分开。静态 `H = 每核心长条上限 × 批大小`，不会整批溢出。
- `dequant`（:290）：int8 tap × 每维尺度 → float，在设备上做；`to_device`、`encode`、`cf_loss` 都会调用，幂等。
- 审阅点：`_audio['ref_reads']` 的簇划分（间隔 >200 格分簇）；`summary` 长歌池化系数 k（静态模式下 `S=256`，超过 8.5 分钟的歌按 2 s×k 池化）。

## 3. 模型前向（`beatfcn/model/p1.py`）

- `encode`（:467）：tap 的 LayerNorm 与频带嵌入一次性完成并转 bf16；行（R/I/U/G/K/Y）拼接；`_mask`（:345）生成稠密布尔掩码（行类可见性 + Y 因果 + K/Y 音频时间窗）；8 层 `Layer.forward`（:92）：联合 softmax，音频 K/V 用 GQA。
  - `packed=True` 走 FlexAttention 块掩码路径（`_encode_packed`、`_block_mask`）。实测比稠密路径慢，默认关闭；等价性测试保留。
- `time_logits`（:512）：指针头 = END + 800 格 + 复制锚点；合法性掩码 `ok`。
- `forward`（:626）：所有损失都通过 `masked_ce`（:155）：按有效行加权，不做 gather，不同步主机；非法目标只计数（`_illegal`），`summarize` 时才报错。
- `_inner`（:673）：内层 GRU 对全部 B×G 槽位稠密计算，固定 `MAX_NOTES+1` 步，无效槽位靠掩码排除。`_hold`（:741）用 `hold_valid`。
- 审阅点：
  - 可见性规则（`_mask` 与 `_block_mask` 必须一致，`tests/test_p1_model.py::BlockMaskTests`）。
  - 泄漏：Y 行只看前缀与因果；I 行只看 R/I；`tests/test_p1_model.py::test_future_groups_do_not_change_earlier_time_logits`。
  - 全程没有 `.item()/.nonzero()/.any()` 数据依赖分支（每小批 2 次同步，均在非图路径的 `index_copy` 以外）。

## 4. 训练步（`beatfcn/train/graph.py`, `beatfcn/cloud/train_app.py:226`）

- `train(...)`：
  - `_select_rows`（:167）用缓存索引 `taps/<key>/_index.json` 选歌；`prep` 参数加载 CPU 预处理的验证片段和配对审计。
  - 多任务配对：`also=[...]` 让多套模型吃同一批数据，各自优化器、存档、日志。
  - 数据：`_stream`（:36）→ 主进程 `next_batch`：预取 `prefetch` 批，`pool.request` 歌曲，`attach_audio`，`pool.release`。
  - 非图路径：`to_device` → 前向 → 反向（支持累积）→ CF（每 `cf_every` 步，即时计算）→ 裁剪 → `opt.step`。
  - 图路径（`graph=True`）：`GraphedStep.fill` 把批拷进静态缓冲区（`w_select` 也是张量）；CF 先即时反向，梯度累加；`GraphedStep.step` 预热 3 步后录制一次，之后每步只重放。优化器 `AdamW(capturable=True, fused=True)`，学习率是张量，`fill_` 更新。
  - 保护：`min_util` + `check_after_s`：占用率低于阈值时所有任务存档退出；`stop_at` 让任务在指定步暂停，调度仍按总步数。
  - 日志：每 `log_every` 步一行 JSON（`data_wait_s`、`gpu_s`、`gpu_util`、池统计、各头损失/准确率）；验证每 `validate_every` 步，存 `best.pt`/`last.pt`。
- `GraphedStep` 的不变量：`_body` 对图与非图完全相同；`tests/test_graph.py` 在 CPU 上验证"非图 GraphedStep == 普通循环"（**未覆盖真实录制**）；图化数值与非图的对比只能在 GPU 上做（`bench_graph` 的 `numerics`：dropout=0、阈值 1e-4）。录制前调用 `pool.wait_idle()`，保证没有其他线程在提交 CUDA 工作。不参与训练的参数保持 `grad=None`，与普通循环一致。
- 审阅点：
  - 录制期间不能有主机同步、不能有随步数变化的 Python 常量（已改成张量的：lr、w_select）。
  - `zero_grad(set_to_none=False)` 在步末，保证静态梯度张量存在；CF 即时反向依赖这一点。
  - 验证 `evaluate` 走非图路径（动态形状），参数在原地被图更新，无需拷贝。

## 5. 显卡常驻歌曲池（`beatfcn/train/pool.py`）

- `request`：后台线程整首读取 → 锁页内存 → 侧流拷到显存；`get`：等加载完成，让当前流等待拷贝事件（无主机同步）；`release` 解除引用；`_evict` LRU 淘汰未被引用的歌。
- 容量 `pool_songs`（默认 160 首 ≈ 17 GB）。引用计数按批次计，`get` 对消费流 `record_stream`。审阅点：**worker 数 × 模式数 × 每 worker 常驻数**（默认 2 × 6 = 12 首/worker）应小于容量，否则反复淘汰重载；`resident_bytes` 不含 pending 和锁页暂存。

## 6. 测试与等价性检查

- `tests/`：108 个单元测试（`python -m unittest discover -s tests`）。
- 关键的"改前后数值一致"检查脚本在会话 scratchpad（`ref_metrics.py`、`packed_eq.py`、`bm_check.py`），比较参考批次上各头的损失/样本数/正确数和梯度范数。改动模型或打包后应重跑：目前基准 `ref_metrics_before.json`（去同步之前）与现版本一致。
- 云端 bench：`bench_train_gpu`（显卡侧每步时间）、`bench_graph`（数据路径 + 图化 + 数值）、`diag_worker`（worker 启动分段计时）。

## 7. 已踩过的坑（审阅时的反例清单）

1. 网络盘（Modal Volume，FUSE）：目录扫描 8.7k 项 + exists 160 秒；mmap 零碎读极慢；每首歌整读要开多个流。→ 索引文件、整读、显卡常驻。
2. 显卡容器启动时做 CPU 工作（验证片段 68 s + 配对审计 54 s + worker 预热）→ `prep_train` 与 CPU 预取。
3. `.nonzero()/.any()/.item()` 每小批 359 次同步 → 全部改成掩码计算。
4. PCA-512 缓存训练效果大幅下降（时间格准确率 34% → 14%）→ 全维度 int8。
5. FlexAttention 打包路径：掩码构建按元素评估（7 亿元素/批）→ 改块级构建后仍比稠密 SDPA 慢 1.7 倍（掩码函数内核内查表）。
6. Phira 和弦 >8 音符在 worker 打包时抛异常使训练崩溃 → 采样阶段跳过并计数。
7. 多容器并行写同一 Volume 时写入争抢，占用率从 95% 掉到 13–88%。

## 8. 还没做 / 已知限制

- 图路径未在 GPU 上验证（`bench_graph` 待跑）。
- `collate_pairs`（反事实配对）仍在 worker 内按 memmap 逐单元读歌（每 4 步一次，量小）；验证每次按 core 重新打开 memmap。两者都还没走歌曲池。
- 样本离线预生成（研究报告排第一的做法）未实现；目前 CPU 侧每份约 0.1 秒，靠加核数扩展。
- `S2`（LoRA）训练循环（`train_s2`）仍是旧数据路径，未改成歌曲池 / 图化；窗口去重按批内下标而非歌曲，冻结的 1–20 层每步重算。
- 断点续训只恢复模型、优化器和步数；采样序列用 `seed + 7919 × 续训步` 换种子，不是精确接续；RNG 状态未保存。
- `prep` 文件只核对行数，不核对清单内容指纹。
