# BeatFCN 原生 osu4K 监督数据桥（本地候选）

这是独立于网络结构的标签转换器，不是 S0b 解码器、训练器或训练结果。
没有修改 shared24+4 S0a 包，没有启动 GPU、上传数据或执行训练。

## 私有数据产物（不随本公开源码分发）

- 32 个不同固定 train 歌曲组，各对应原清单中的一个 12 秒请求。
- 32 个 requests/*.json.gz，合计约 0.84 MB。每个文件分开保存合法请求、参考证据、监督标签和来源记录。
- 2,768 个新增音符目标，其中 316 个 hold，组成 1,983 个 onset group、58 个核心片段。
- 53,314 个不可修改的原事件仍保留在请求中，包括远处与未来锚点。没有为了前缀容量而删除原事件。
- 37 个与模拟明确留白相交的隐藏对象整枚移除，逐个记录原因。不是剪短长条，也不是把自然缺音当负例。
- 原文件和音频仍从既有本地源读取，并核验 SHA256；不把音频复制进此包。

这些是人类谱师原谱导出的真实时间、类型、键道和长条端点，但任务掩码是程序构造的。
原谱不是新取得的真实半成品编辑会话；所有样本都未经过独立人工听音对齐审阅，gold=false。
现有来源的训练/再分发权利仍未核实。本包不解除后续训练或转移的权利门槛。

## 准确支持的任务子集

按清单顺序固定循环分配五种构造任务，不按隐藏音符数量选择掩码边界：

1. middle_gap，7 个：12 秒待补区域，两侧完整原谱作为不可修改上下文。
2. sparse_anchors，7 个：待补区每 2 秒内固定前 300 ms 的已有对象保留为锚点；其余空处未知。稀疏参考仅提供正例。
3. complete_reference，6 个：前 4 秒作完整参考，后 8 秒待补；参考过密时只缩短完整支持区间，不丢弃区间内的音符。
4. multiple_gaps，6 个：前 4 秒和后 6 秒可编辑，中间 2 秒完整且受保护。
5. explicit_blank，6 个：12 秒中第 5–6 秒明确留白，与其相交的隐藏对象整枚去除。

所有权限分区覆盖整首音频，使用左闭右开区间。跨入受保护区域的 hold 整枚放入 observed，绝不裁短或留下无法合法生成的目标。若模拟留白与原先锁定的外部 hold 冲突，该请求应标为 unsupported，而不是改人谱；本批没有遇到该冲突。

complete 参考的“完整”是本构造配方对完整发表原谱的覆盖假设，并有原文件哈希与全量参考事件检查，不是人工重新认证。只在此类参考内部构造 unselected，含义仅为“此原谱未选中该时刻”。候选来自确定性 40 ms 非音符网格，排除头部 ±30 ms、活动长条和边缘 30 ms；它不是规格中的一半 spectral-flux 峰值采样器，不是乐器、声部、节拍或反事实意图标签。当前 357 个 selected、258 个 unselected 只用于明确声明的参考条件。

## 保真、监督与防泄漏

- 原生 osu 时间是相对音频的毫秒，不加 AudioLeadIn，不估 BPM，不自动 Redline，不对齐算法峰值。正 uninherited timing 与负 inherited 控制完整保留为来源证据；100% 请求对模型隐藏精确 timing map。
- 原始文件字节是 immutable sidecar，native_ref 指向 HitObjects 原始序号。事件 ID 使用完整源 SHA256 加原始对象序号；同一原文确定性稳定。相同时间同键碰撞直接隔离，不合并、不随机改 ID 掩盖问题。
- 原生事件不量化。只有隐藏新增事件的模型视图按 1 ms half-up 规则量化后分组；与已知锚点精确同刻时使用 COPY_TIME。同键碰撞、非正 hold 或量化越界均拒绝。此批源对象本身为整数毫秒，不代表已验证 1 ms 听音精度。
- 核心最多 8 秒，超组数/事件数上限按时间继续细分，不丢标签。组内按键道排序；非空组以 END_GROUP 结束，每个待补片段显式 END_GAP。长条端点始终完整，可超过当前核心；后续核心仅接收过去生成的活动 hold，不再生成其头。
- model_condition 是独立白名单投影，不读取 supervision 或来源记录。没有隐藏目标、真实数量、文件名、歌曲组、谱师、原生事件序号、chart hash 或 oracle timing map。audio_id 只作音频读取路由键，不得嵌入为歌曲身份特征。
- teacher_forcing_steps 将 input 与 target 分开：仅完成的过去组可反馈，不给当前组真值数量；未来锚点只能来自 observed。endpoint_teacher_forcing_fields 的 block 阶段看不到 end_us 或真值 block，之后的条件步骤才允许读取已选 block 细节。
- 参考池及其原始覆盖证明在审计侧保存；模型侧仅接收合法已公开事件及 selected/unselected。editable 中未出现音符不产生负意图标签，selection BCE 和 CF 标签均为 None。

这些是数据层隔离和可验证的条件序列，不是对某个网络注意力掩码、梯度、缓存或解码实现的证明。

## 运行

Python 3.10+；转换和语义验证只用标准库。完整 JSON Schema 检查使用环境已有的可选 jsonschema；缺少时测试会明确 skip，不会谎称通过该项。

    cd <experimental_data_directory>
    python -m beatfcn_data build --source <source_root>
    python -m beatfcn_data validate
    python -m unittest discover -s tests -v

默认 loader 同时核验原图谱、音频、原 chart-relative 音频引用、固定 split 行和输入清单哈希。缺源或哈希不符会失败，不降级造标签。

    from beatfcn_data import load_artifact, model_condition, teacher_forcing_steps
    a = load_artifact('requests/train-human-reference-01.json.gz')
    condition = model_condition(a)
    for step in teacher_forcing_steps(a):
        model_input = step['input']
        loss_target = step['target']
        # 这两个对象必须分开；本包没有模型调用或训练循环。

resolve_native_sidecar 和 restore_original_bytes 可读取哈希核验过的原始行/字节；它们不导出生成谱，也不重写原谱。

## 未完成与不支持

- 未实现 Phira、全曲从零滚动生成、完整前缀续写课程、真实用户编辑会话、独立人工对齐 gold、原生 renderer golden tests。
- 未实现真实同曲 A/B 非嵌套反事实配对、选音 BCE、完整 spectral-flux 参考采样、控制统计/分档及 dropout。controls 故意缺省，不泄露精确局部数量。
- 未实现事件特征编码、group GRU、COPY_TIME 推理、端点预测器、约束采样器、导出器、损失函数、训练、32 样本过拟合或质量评测。
- 没有把本包直接接入 S0a 的合成 8 维 feature 接口。全曲 observed 索引需要另外实现无目标泄漏的局部上下文/远程摘要与资源预算，不能直接塞进 1536-token 前缀后静默截断。
- 32 请求只覆盖 osu4K，不能宣称满足规格要求的两模式完整 S0。当前架构仍由 shared24+4 S0a 合同决定，本桥没有恢复历史独立事件 Transformer 架构。

原始构建的私有报告记录了本地验证；这些报告与请求数据不在公开源码中。根任务应在独立审查后决定是否集成；转换成功不等于允许训练、允许上传或已经学会人类采音。

## Public source handoff status

Prototype only, NOT integrated. Independent review pending. The original local build passed 30 tests, including real-source/schema checks. This public source package intentionally omits private artifacts and source material, so real-artifact integration tests skip until an authorized local dataset is supplied. Core synthetic tests remain executable. No audio, original charts, private request artifacts or local machine paths are included. The publication copy was then rerun: 17 synthetic/source-only tests passed and 13 private-fixture tests skipped; with the separately held private labels plus original source root, all 30 tests passed again, including schema validation. This is execution evidence, not an independent implementation review.

## Relocatable private-fixture verification

The labels and original pilot data are delivered separately in the task conversation. Restore `beatfcn_private_32_labels.zip` to a private labels directory and the original pilot to its own source root. Preserve the fixed 71 train / 10 validation / 19 test split; all 32 requests must remain train-only. Do not publish either directory or read held-out bodies into training.

From this source directory:

    python -m unittest discover -s tests -v
    python -m beatfcn_data validate --root /absolute/path/private_labels --source-root /absolute/path/original_pilot
    BEATFCN_TEST_ARTIFACT_ROOT=/absolute/path/private_labels BEATFCN_SOURCE_ROOT=/absolute/path/original_pilot python -m unittest discover -s tests -v

Without private artifacts, the source-only suite deliberately skips the real-data integration class. The two environment variables only select local test input roots; no paths, datasets or credentials are uploaded. Optional jsonschema must be installed to include the schema check. Source fields are verified against original hashes. This bridge is still not a model, trainer, counterfactual-gold dataset, or integrated S0b implementation.
