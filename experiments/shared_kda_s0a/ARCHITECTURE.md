# Shared Pupu24 plus full KDA4 architecture

This is a numerical scaffold, not a completed chart-writing model.

## Fixed architecture and exact actual tensor counts

- Native Pupu patch embed +24 pretrained layers, d1024,16 heads, preserved
  2D audio RoPE. Strict294-tensor loading,403,141,632 frozen parameters.
- Audio patches AND right-shifted event-group feature tokens traverse the SAME
  QKV, projection, normalization and SwiGLU weights in ALL24 layers. Audio is a
  frozen, cacheable stream; trainable rows read its K/V at every layer. There
  is no independent40M event Transformer.
- Four NEW FULL serial blocks after6/12/18/24: pre-RMSNorm +8head×128 KDA
  residual +pre-RMSNorm +SwiGLU4096 residual. Each17,328,264 trainable params;
  all four69,313,056. This includes50,331,648 new FFN parameters.
- Small new feature/intent/position/selection/probe heads:4,285,229 params.
- **Total476,739,917; trainable73,598,285; frozen403,141,632.**
  BF16 weights953,479,834 bytes; FP32 weights1,906,959,668 bytes.
  These are storage counts, NOT peakVRAM or a4GB feasibility claim.

`reports/parameter_inventory.json` is computed from actual instantiated meta
parameters, not a paper/model-card size label. Base publication revision:
`c0f9345a1de0e61237dec7da32a70049c4ec0fa8`. Local recovered encoder subset
SHA256:`6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4`.
Original full-publication-file hash was NOT verified by this recovered subset.
Weights/audio are intentionally not copied into this package.

## KDA definition and source boundary

Independent PyTorch reference follows published Kimi Delta Attention, not
ordinary linear attention: q/k/v projections ->causal depthwise conv4 ->SiLU;
L2 q/k; key-channel decay first; beta delta correction against the decayed
state; q readout with1/sqrt(head_dim); per-head RMSNorm and sigmoid output gate.
Alpha uses low-rank128→1024 gate, softplus, learned per-head exp(A_log), and
per-key dt_bias. Beta is sigmoid, NOT doubled. Value is NOT L2 normalized.
State is FP32[B,H,K,V] (float64 in equation test).

Chosen S0 config: qkv/output/beta/lowrank projections bias-free, conv biases
included, output norm affine, L2 eps1e-6, RMS eps1e-6, A_log initialized0,
dt_bias−3, conv normal std0.5, default PyTorch linear initialization. These are
explicit engineering choices. This is NOT bit-identical to the complete released
Moonshot/FLA module or its initialization. No checkpoint for new KDA is claimed.

Primary source parameterization:
https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/modeling_kimi.py
(MIT; revision verified from model API).
Reference recurrence and optional kernel:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py
(MIT; moving main reviewed; official PyPI fla-core0.5.2 wheel now pinned,
SHA256-verified and API statically inspected; NOT installed or GPU-tested).
Paper: https://arxiv.org/abs/2510.26692 .

`backend='fla'` calls the actual optional `fla.ops.kda.chunk_kda` kernel with
already-normalized q/k and transformed gates/beta, state_v_first=False. It
never silently switches to generic linear attention. CUDA/FLA forward-backward
parity has NOT been run. Use the verified `requirements-fla.txt` pin and separately pin a compatible
CUDA PyTorch>=2.7/Triton>=3.3/einops environment; run `scripts/check_fla.py`
before using that backend. Its successful T16 and cross-chunk T129 forward/state/backward report is
mandatory, matched to GPU/Torch/CUDA/Triton/BF16 and local KDA source hash. The pinned wheel signature matches the wrapper; runtime kernel/import/math
compatibility remains a pre-GPU gate. Incremental inference cache is explicitly reference-only.

## Visibility, frozen audio stream and recurrent-state schedule

Revision 2026-10-22b ("frozen audio memory"). Order is A (local music
patches), G (global music summaries), R (paired reference features), I (8
learned intent tokens), W (known/control features), E (BOS then completed
previous-group attributes). Boolean SDPA edges:

- A reads valid A only. A rows receive NO modality embedding and NO KDA
  residual, so at every layer they are exactly the pretrained Pupu forward
  (tested bit-exact against `FrozenLargeTeacher`). They never read G/R/W/E.
- G reads A/G. R reads R only. I reads R/I only. W reads A/G/R/I/W, not E.
- E reads all legal prefix and its own/past shifted positions. Position0=BOS
  predicts group0; positioni embeds groupi−1 and predicts groupi.

All 24 shared layers still serve both streams with one QKV/projection/norm/
SwiGLU parameter set: audio patches in the frozen stream, G/R/I/W/E rows in
the trainable stream reading audio K/V of the SAME layer. Change from the
first S0a revision: audio rows no longer get a trainable modality vector, no
longer read G, and no longer receive KDA residuals. Trainable audio
adaptation remains through each KDA block's ordered scan over audio, which
seeds every G/W/E row.

### AudioMemory: the reusable cache

Because A depends only on (mel, audio_valid, frozen backbone), it is computed
once under no_grad (`SharedModel.encode_audio`) and stored as per-layer
post-RoPE K/V, the hidden entering each KDA insertion, and the final
LayerNorm hidden. The same object is reused:

- across training steps: trainable updates never stale it (tested by an
  update-then-compare check), and A needs no backward or stored activations;
- across requests/edits on the same window: edits to O/R/controls/scene only
  rerun the trainable rows (hundreds of tokens, not 2048 audio patches).

Its key hashes mel bytes, valid mask and backbone parameter identity/version/
dtype/device. A different mel, mask or ANY backbone change (load, future LoRA,
in-place edit) is rejected as `stale audio memory`, never silently reused.
Future S2 LoRA on layers 21–24 must also rebuild layers ≥21 of AudioMemory.

Size per 10.24 s window, BF16: 222,298,112 bytes (24 layers K/V + 4 taps +
final). This suits an in-memory LRU per song bucket, not a corpus-wide disk
cache (≈35 windows per 3-min song). At corpus scale training recomputes the
frozen stream under no_grad per song bucket and reuses it across that song's
requests.

### KDA: two scans instead of five

Each KDA block runs two shared-weight scans per example: (1) R then I,
emitting R and I rows; (2) the audio-prefix state, then G,R,I,W,E, emitting
G, W and E rows. A causal scan's output at a row depends only on earlier
rows, so this equals the per-class routes R; R→I; A→G; A→G→R→I→W;
A→…→E (tested against independently computed routes). R/I never see audio,
G, workspace or targets. The audio-prefix state (`KDABlock.audio_prefix`) is
recomputed with gradients in training and memoized in
`AudioMemory.kda_prefix` only when autograd is off, keyed by that block's
parameter identity/version and model revision.

Per-example valid positions are compacted before recurrence, so masked tokens
advance neither conv nor recurrent state. Batch examples start fresh. Cached
decode steps retain each layer's K/V (audio + prefix + events), valid-key mask
and each KDA event state. Cache reuse checks exact context hash, explicit
revision AND parameter object/version fingerprints; edits require re-prefill,
which with AudioMemory costs only the trainable rows.

`backend='chunk'` is an exact chunked KDA in plain PyTorch (CPU/MPS/CUDA):
per chunk u=(I+A)^-1 diag(β)(V−K⁺S₀) via the nilpotent doubling product,
decays only as exp(G_t−G_j≤0). It matches the step recurrence to ~1e-15 in
float64 for outputs, final state and all gradients, keeps only chunk-boundary
states for autograd, and is what makes the native window trainable outside
CUDA/FLA. Routing uses index_select/index_add because advanced-index backward
lacks BF16 on MPS (torch 2.6).

Native audio gets native 2D RoPE; G and other tokens get identity RoPE. Event
index has learned embeddings and continuous attributes have an MLP. This
positional choice is a scaffold, not trained alignment.

S0 local input is ONE bounded native≤10.24s normalized-mel window; no 16 s
stitching, center-retain overlap policy or remote retrieval is implemented.
Global music accepts≤600 explicit audio-only feature tokens, visible to G/W/E,
including future music. Reference/known/global-token prefix≤1536; local raw
patch tokens are counted separately. Excess raises resource_exhausted, never
crops labels. A tiny selection readout queries isolated intent from audio
only, then averages local scores across 8 frequencies into 25 Hz cells. It is
untrained, not strict CF supervision or evidence of learned human selection.
Hidden labels must NEVER be supplied as known features.

## Autograd

Original pretrained parameters are frozen. The audio stream runs under
no_grad (it has no trainable inputs); trainable-row forwards through the same
frozen layers keep autograd so gradients reach event/intent embeddings and
every KDA block, including through the KDA scan over audio. No LoRA in S0a.

## What actually ran

See VERIFICATION.md for the frozen-audio-memory revision (28 CPU tests and the
native-window MPS probe). The original first-revision record follows.


The core CPU suite contains 11 tests, reduced-width d32 BUT24 shared layers and4
full KDA insertions. Tests cover actual full-size meta inventory; independent
matrix-equation forward/backward; conv+recurrent step parity; shared QKV identity
for audio/events; all-new-parameter nonzero finite gradients; forbidden
future/intent Jacobians; hidden-future perturbations; padding/batch reset;
teacher-forced vs cached decoding; context/model/weight invalidation; global
music reachability; strict-loader missing-key rejection; toy save/reload parity.

`reports/real_weight_cpu_probe.json`:actual full476.74M model, strict294-tensor
checkpoint load and verified local SHA256;40ms real local audio excerpt plus
synthetic reference/known/global/previous-group features; BF16 forward/backward
and one explicit SGD update. All trainable modules have finite gradients,
frozen gradients absent, sampled frozen values unchanged. CPU peakRSS≈2.70GB.
This is not a real supervised chart task, a full native-window performance
measurement, GPU benchmark, full frozen-tensor posthash, or audio quality result.
Only a short local technical smoke; training/redistribution rights unverified.

