# Verification scope

## Previously completed numerical evidence

- Eleven core CPU tests cover reduced-width but 24-shared-layer/four-full-KDA graphs, actual full-size meta parameter inventory, independent KDA forward/backward equations, streaming recurrence, gradient reachability, leakage isolation, padding/reset, cached decoding and invalidation, and strict loading/save/reload.
- Eleven launcher CPU tests cover local-only help/plan/import, explicit stage authorization, exact limits, billing checks, source/input allowlists, endpoint restriction, symlink rejection, and stopping after failed parity.
- Static Modal 1.6.2 API option construction was checked with socket connections disabled. This did not enter app.run, build, spawn or remote execution. See `reports/modal_sdk_static_check.txt`.
- The full-size real-weight CPU probe loaded 294 tensors and performed one BF16 update on 40 ms of real audio plus synthetic conditions/targets. See `reports/real_weight_cpu_probe.json`. That earlier audio fixture was not the new Higher GPU candidate. The report is not a native-window benchmark or supervised chart result.

## Frozen audio memory revision (2026-10-22b)

- CPU suite now 28 tests. Added: chunked KDA equals the step recurrence (float64, outputs/final state/all gradients, padding, initial state, strong decay); two-scan routing equals independently computed per-class routes; local audio stream bit-exact to the pretrained `FrozenLargeTeacher`; AudioMemory has no autograd, survives an SGD update bit-exactly, is reusable across workspace/reference edits, and is rejected after mel or backbone change; KDA audio-prefix memo is invalidated by KDA weight change and bypassed under autograd; chunk backend matches reference for full forward and cached decode. Launcher tests now resolve the real temp dir (macOS `/tmp`/`/var` are symlinks; the symlink guard itself is unchanged).
- Full-size native-window probe on Apple GPU (MPS, BF16, synthetic audio/events, `--backend chunk`): see `reports/mps_native_probe.json`, including a same-shape timing comparison against the first S0a revision. MPS has no peak counter; memory figures are allocator samples at phase boundaries.
- Not established by this revision: CUDA/FLA parity (FLA path now receives an initial state from the audio prefix; still CUDA-gated and untested), real Higher audio, SFT, quality, 4 GB feasibility.

## Public-package checks

The sanitized public copy is rerun through the complete CPU suite and local help/plan before publication. `reports/public_cpu_tests.txt` records that result. Only source/docs/tests and sanitized technical evidence are included. The source manifest records each published file's SHA256; it intentionally excludes itself.

The public teacher-fetch header was also reconstructed offline and compared byte-for-byte with the audited local subset header. Its 294 tensors and exact 1,612,597,160-byte total matched; no network download was performed. See `reports/teacher_fetch_offline_check.json`.

## Not established

No CUDA execution, FLA runtime parity, Modal upload/build/run, native-window memory/timing, SFT convergence, music quality, 4 GB feasibility, full frozen-tensor post-update hash, dataset alignment gold status or broad training/redistribution rights. Static review and CPU tests do not establish those outcomes. Do not reinterpret the earlier CPU report as a Higher or GPU result.
