# Local-agent handoff: shared Pupu24 + full KDA4 S0a

Date: 2026-10-22. Scope: reproduce local checks, prepare exact inputs, then conduct one explicitly authorized, bounded Modal numerical probe. Do not substitute the repository's old v24 training scripts or a separate frozen-audio + 40M event writer.

## 1. Current status and settled architecture

This directory is additive and isolated. The repository's existing `main` files and older model implementations are preserved.

- One shared PupuM2D Large encoder: patch embedding + 24 pretrained layers, width 1024, 16 attention heads, native 2D audio RoPE. Its 294 tensors / 403,141,632 parameters are frozen, but the shared forwards retain autograd.
- Both audio patches and right-shifted event-group feature tokens traverse the same QKV, projection, normalization and SwiGLU weights in all 24 layers.
- Four new **full** serial KDA blocks after layers 6, 12, 18 and 24: pre-RMSNorm + 8 x 128 KDA residual, then pre-RMSNorm + SwiGLU-4096 residual. These are not attention-only inserts.
- Total 476,739,917 parameters; trainable 73,598,285. Four KDA blocks account for 69,313,056 trainable parameters; feature/intent/selection/probe heads account for 4,285,229. No LoRA in S0a.
- The scaffold implements visibility-isolated attention and two shared-weight KDA scans per block (R→I; audio prefix→G→R→I→W→E), exactly equal to the five per-class routes of the first revision. It does not pretend a causal recurrence can implement an arbitrary mask. See `ARCHITECTURE.md`.
- Revision 2026-10-22b, frozen audio memory: audio rows read only audio and receive no modality/KDA residual, so every layer equals pretrained Pupu. `SharedModel.encode_audio` builds an `AudioMemory` once under no_grad; it is reused across training steps and edits and rejected if mel, mask or any backbone weight changes. Trainable rows (G/R/I/W/E) still traverse all 24 shared layers and read audio K/V at each layer.

Passed: 28 CPU tests (22 original, adjusted for the frozen audio stream, plus chunked-KDA exactness, two-scan route equivalence, bit-exact Pupu audio stream, AudioMemory reuse/invalidation, KDA prefix memo, chunk backend decode); static Modal SDK 1.6.2 option construction without sockets; strict full-size checkpoint loading and one short CPU BF16 forward/backward/manual-SGD update (historical, 4 mel frames). New: full-size native 10.24 s BF16 forward/backward/manual-SGD on Apple GPU (MPS) with `--backend chunk` and synthetic audio/events; see `reports/mps_native_probe.json`. None of this is SFT, learned chart quality, or proof of 4 GB feasibility.

Not run: Modal upload/build/CPU preparation/GPU execution, CUDA/FLA numerical parity, CUDA native-window peak memory or runtime, the Higher real-audio excerpt (private input not present on the operator machine yet). Use only a legitimate supported connection on the operator machine; do not bypass an access denial.

## 2. Inputs: never commit their bytes

The source repository/ZIP contains no audio or model weights. Resolve files on the consuming machine, verify their contents there, and use explicit local paths. A path on another executor is not a transfer.

Teacher subset:

- Filename: `teacher_encoder.safetensors`.
- Exact size: 1,612,597,160 bytes.
- SHA256: `6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4`.
- Exactly 294 patch-embedding/EMA-teacher tensors, 403,141,632 parameters.
- Pinned publisher revision: `c0f9345a1de0e61237dec7da32a70049c4ec0fa8` in [spellbrush/PupuM2D](https://huggingface.co/spellbrush/PupuM2D/tree/c0f9345a1de0e61237dec7da32a70049c4ec0fa8).
- The full publisher file's LFS SHA256 is recorded as provenance, **not verified** by this subset. Do not equate range/subset integrity with verification of the entire 3,363,632,912-byte publisher file.

If the exact subset already exists locally, use it. Otherwise the included `teacher_fetch/src/download_teacher.py` reconstructs the same subset with Python standard library and curl, using the pinned public source and preserved safetensors header. It downloads about 1.61 GB on the operator machine, not in the GPU worker. Have at least about 2.7 GB free before starting; do not import this script because it executes at top level. Do not change or reformat its header file: its byte length participates in source offsets.

```sh
mkdir -p teacher_fetch/weights
python teacher_fetch/src/download_teacher.py
```

Do not rerun it over an existing valid file. After download, the manifest step below enforces the exact required hash and size. A mismatch is a stop condition, never permission to accept different weights.

Higher audio:

- Obtain the separately delivered private input ZIP from the task conversation. Do not upload it or its download link to GitHub.
- Extract `higher_audio.mp3` locally. The filename extension is historical; the actual original stream is Vorbis and is decoded by ffmpeg. Do not transcode it because the launcher requires the original bytes.
- Exact size: 4,128,524 bytes.
- SHA256: `ffc5a731ccd5797e81e95fb75a8f575dc92c9cca6ab8116a17a8b1c65582cd98`.
- Chart identity: osu!mania `4067533`, Higher / FIFTY FIFTY / fvrex / Easy; dataset revision `b82117a5de13b7df5290482aed7cffc49cb76e6d`. Public provenance is in `contracts/modal_probe_candidate.json`.
- The numerical probe starts at **15.00 s** and uses **10.24 s**, ending at 25.24 s. Do not silently select the initial silence or substitute a recording.
- The optional chart in the private ZIP is provenance only; the Modal allowlist uploads only the teacher subset and audio. No human-note chart supervision is used in S0a.
- Community authorship is source-reported. Independent timing alignment and broad training/redistribution rights have not been verified. This is not gold timing data.

Keep credentials, billing evidence, state ledgers, audio, charts and weights outside public commits. `.gitignore` is a guard, not an authorization to publish anything not ignored.

## 3. Local checks

Run from `experiments/shared_kda_s0a`. Local orchestration requires Python 3.11+ on macOS/Linux and Modal SDK 1.6.2; it does **not** need local CUDA, PyTorch or FLA.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-modal.txt
python scripts/modal_launch_s0a.py --help
python scripts/modal_launch_s0a.py plan
python -m unittest discover -s tests -p test_modal_launcher.py -v
```

To run the full 22-test CPU suite, use an environment with PyTorch, NumPy and safetensors. The original checked environment was Python 3.12 / torch 2.6.0+cpu. For a compatible platform, install the pinned test dependencies and run:

```sh
python -m pip install 'torch==2.6.0' 'numpy==2.2.6' 'safetensors==0.5.3'
python -m unittest discover -s tests -v
```

Do not install `requirements-fla.txt` into the CPU test environment. FLA/CUDA dependency installation belongs to the explicitly pinned remote image.

Optional local CPU full-size smoke, only if sufficient RAM and ffmpeg are already available:

```sh
python scripts/probe.py --checkpoint /absolute/path/teacher_encoder.safetensors --device cpu --audio /absolute/path/higher_audio.mp3 --audio-offset-seconds 15 --mel-frames 4 --groups 2 --steps 1 --output reports/probe_cpu.json
```

That remains a numerical smoke with synthetic heads, not training or quality evaluation.

Native-window probe on Apple Silicon (MPS, no CUDA/FLA, no cost). Uses the exact chunked PyTorch KDA; FLA parity remains a CUDA-only gate:

```sh
python scripts/probe.py --checkpoint /absolute/path/teacher_encoder.safetensors --device mps --backend chunk --mel-frames 1024 --groups 2 --steps 2 --output reports/probe_mps_1024.json
```

Add `--audio /absolute/path/higher_audio.mp3 --audio-offset-seconds 15` once the private input is available. On macOS, keep manifests/billing records outside `/tmp` and `/var` (both are symlinks; the launcher rejects symlinked paths by design).

## 4. Authorized bounds and required preflight

The bounded experiment has a **$1 actual-usage target, $2 gross-usage ceiling, and $0 card charges**. Include build, startup, CPU, GPU, memory, storage and egress. Neither a client timeout nor the printed estimate establishes a hard invoice cap.

Before any remote action, use the already authorized Modal account/profile and perform a current read-only billing check. Verify applicable credits, the workspace-wide gross cap at or below $2, net spend limit $0, and other running workloads. Do not change account settings, create credentials or increase spending permissions. If the verified configuration cannot satisfy these bounds, stop and ask the operator. A copied old billing record is not current evidence.

Copy `reports/modal_billing_record.template.json` to an operator-local JSON file, fill only truthful non-secret evidence, and set clearance only after the check. It must be no more than 15 minutes old at each remote stage. The template deliberately fails closed. Store the filled record outside version control. The record and flags document existing authorization; they do not grant it.

Remote runtime: Python 3.12, torch 2.7.1+cu126 from the official CUDA 12.6 PyTorch index, Triton 3.3.1, fla-core 0.5.2 with SHA256 enforcement, einops 0.8.1, NumPy 2.2.6, safetensors 0.5.3. This is not a fully hashed OS/transitive dependency lockfile.

Hard design bounds:

- One L40S, one container, two CPU cores and 16 GiB RAM with matching hard limits.
- 600-second GPU function timeout, 60-second startup allowance, 2-second scaledown, no warm pool and single-use containers.
- Shared absolute 900-second session deadline beginning before upload; upload watchdog 180 seconds, CPU/image preparation watchdog 240 seconds, internal GPU subprocess deadline at most 580 seconds.
- No automatic application retries, smaller-model retry, reference-backend fallback, full SFT, Adam state, deployment or public endpoint.
- Only `https://api.modal.com`; no alternate endpoint bypass.
- FLA BF16 forward/state/backward parity at T=16 **and** T=129 must pass before starting the full model.
- Only then: 1024 mel frames / 10.24 s, batch one, two synthetic previous-event groups, one BF16 forward/backward/manual-SGD update, exact 476,739,917-parameter model.

## 5. Exact launch sequence

Replace paths below with files that exist on this machine. Do not run GPU stages unless the current authorization and preflight checks above are satisfied. Keep all stages in the same session and source revision. The default run ID is `s0a-higher-001`.

```sh
python scripts/modal_launch_s0a.py create-manifest --checkpoint /absolute/path/teacher_encoder.safetensors --audio /absolute/path/higher_audio.mp3
python scripts/modal_launch_s0a.py upload-assets --approved-upload --billing-record /absolute/path/current-billing-record.json
python scripts/modal_launch_s0a.py prepare --approved-upload --approved-compute --billing-record /absolute/path/current-billing-record.json
python scripts/modal_launch_s0a.py probe --approved-upload --approved-compute --billing-record /absolute/path/current-billing-record.json
python scripts/modal_launch_s0a.py retrieve
```

Inspect each stage's result before proceeding. Do not execute this list blindly as an unconditional script. `create-manifest` is local only and must pass the input/source hashes. Remote stages reserve their attempt in `reports/modal_session.json` before external calls. Do not delete/reset that ledger or create a new run ID to evade a failed or uncertain attempt. If a stage failed or its outcome is uncertain, inspect retained state/provider status and stop instead of repeating it.

Expected retrieved evidence: `prepare.json`, `runtime.json`, `fla_parity.json`, `cuda_fla_native_probe.json`, `launcher_status.json`, `probe.log` under `reports/modal_result`. Record missing artifacts explicitly; retrieval completion alone does not mean the probe passed. Preserve versions, hashes, actual parameters, gradients, peak CUDA allocated **and reserved** memory, timing, status and actual billing evidence.

After retrieving the available evidence, deletion requires explicit approval for this run's temporary assets/results:

```sh
python scripts/modal_launch_s0a.py cleanup --approved-cleanup
```

Until cleanup is verified, storage remains retained. Empty volume metadata and image-cache metadata may remain even after file cleanup.

## 6. Stop conditions and report back

Stop on failed local tests; input hash/size mismatch; missing authorization or current full-scope billing evidence; denied API access; unexpected terms/permissions; dependency or source mismatch; parity failure; non-finite or missing gradients; frozen gradients; wrong parameter count; OOM; timeout; any failed/uncertain stage; or any change that can exceed the bounds. Do not improvise by weakening checks, increasing memory/GPU/count/time, changing architecture or repeating paid work.

Return: exact source commit, local tests, each remote stage's executed/not-executed status, parity result, model result and parameter count, peak memory, runtime, actual costs if available, files retained/cleaned, and exact blockers. Do not call S0a successful unless both parity and the full native-window numerical probe pass. Even success establishes only numerical feasibility, not SFT quality or a usable chart writer.

## 7. Next-stage data and product work, not part of this launch

A separate pilot selected 32 distinct fixed train-song/audio requests with 12-second intervals; 71 train charts passed structural checks. Validation/test data were not used for training. These are source-reported human community charts, not independently alignment-audited gold data. The separate [native data bridge](../native_data_bridge/README.md) is now included as source only in `../native_data_bridge`. It is an **experimental prototype, not integrated with S0a, with independent implementation review pending**. Its publication copy passes 17 source-only tests with 13 private-fixture skips; using the separately delivered private labels and original source root, all 30 tests pass, including schema validation and relocated source verification. No requests, corpus, audio or original charts are included in the public source. This does not establish counterfactual gold data, SFT readiness or chart quality.

Before S0b, complete the canonical event/group encoder, immutable observed/native sidecars, actual reference builder, strict split isolation and semantic validation, leakage tests and real supervised losses. The current eight-dimensional features and output heads are synthetic probes. Missing model/product pieces include group GRU, COPY_TIME, hold endpoint block/reveal/pointer, Phira location distribution, legality constraints, strict counterfactual pairs/loss, real global audio summaries/retrieval, SENet timing integration, chart export, 32-request memorization, efficacy evaluation and editor integration. Do not launch full training or claim these are already implemented.

### Using the separately delivered data inputs

`beatfcn_private_32_labels.zip` contains the 32 relocatable request/label artifacts. The original pilot is delivered separately as `osu4k_pilot_100_complete.tar.gz` or its split parts. Restore the original pilot layout and verify the archive/part hashes provided with that delivery before use. The fixed corpus split is **71 train / 10 validation / 19 test**; all 32 requests are train-only. Never promote held-out data into training.

The bridge README documents exact validation and test commands using explicit local `--root`, `--source-root`, `BEATFCN_TEST_ARTIFACT_ROOT` and `BEATFCN_SOURCE_ROOT` paths. The paths select local inputs only and do not authorize upload or GPU work. Do not place private data in Git commits. The S0a Modal launch remains unchanged and still uses only the Higher audio and teacher subset with synthetic event heads.
