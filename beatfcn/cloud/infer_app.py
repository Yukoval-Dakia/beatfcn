"""Modal app: generate a whole osu!mania 4K chart for one audio file with a trained P1 checkpoint.

  .venv/bin/modal run beatfcn/cloud/infer_app.py::main --run-id s1-final --audio /path/song.mp3 --out song.osu

From scratch (O=[], all editable). Cores of 8 s are generated left to right; finished cores become
"already generated" context for the next one (design §4.4). Times are model-view times, which
include the osu source constant c (DATA_RULES v2); the export subtracts it. The model does not infer
tempo: the file carries ONE placeholder timing point and says so in its metadata.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DATA, WEIGHTS = '/data', '/weights'
OSU_C_MS = 14  # DATA_RULES v2: osu charts sit 14 ms early relative to decoded audio
app = modal.App('beatfcn-infer')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)
weights = modal.Volume.from_name('beatfcn-weights', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12').apt_install('ffmpeg')
         .pip_install('torch==2.14.1', index_url='https://download.pytorch.org/whl/cu130')
         .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3', 'jsonschema==4.26.0')
         .add_local_python_source('beatfcn'))


def _osu_text(title, notes, version, audio_name):
    lines = ['osu file format v14', '', '[General]', f'AudioFilename: {audio_name}', 'AudioLeadIn: 0', 'Mode: 3', '',
             '[Metadata]', f'Title:{title}', f'TitleUnicode:{title}', 'Artist:unknown', 'ArtistUnicode:unknown',
             'Creator:BeatFCN', f'Version:{version}',
             'Tags:beatfcn generated placeholder-timing (timing point is a placeholder; re-time in the editor)', '',
             '[Difficulty]', 'HPDrainRate:7', 'CircleSize:4', 'OverallDifficulty:7', 'ApproachRate:5',
             'SliderMultiplier:1.4', 'SliderTickRate:1', '', '[TimingPoints]', '0,500,4,2,0,40,1,0', '', '[HitObjects]']
    for t_ms, lane, end_ms in sorted(notes):
        x = int(512 * lane / 4 + 256 / 4)
        if end_ms is None:
            lines.append(f'{x},192,{t_ms},1,0,0:0:0:0:')
        else:
            lines.append(f'{x},192,{t_ms},128,0,{end_ms}:0:0:0:0:')
    return '\r\n'.join(lines) + '\r\n'


@app.function(image=image, gpu='H100', volumes={DATA: data, WEIGHTS: weights}, cpu=4.0, memory=32768, timeout=3600)
def generate(run_id: str, audio: bytes, pca: str = 'pca512_9-16-24', greedy: bool = False,
             temperature: float = 0.8, seed: int = 0, controls: dict | None = None):
    import subprocess
    import tempfile
    import numpy as np
    import torch
    from beatfcn.audio import memory as M
    from beatfcn.audio.pupu import load_teacher
    from beatfcn.audio.preprocess import waveform_mel
    from beatfcn.events.canonical import Event, LaneLoc, build_model_view
    from beatfcn.events.requests import TrainingSample
    from beatfcn.model import features as F
    from beatfcn.model.batch import SongCache
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.sample import generate_core, core_context
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix='.audio') as f:
        f.write(audio)
        f.flush()
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f.name, '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'],
                             capture_output=True, check=True).stdout
    wave = np.frombuffer(raw, dtype='<f4').copy()
    wave = wave[: len(wave) // 960 * 960]
    mel = waveform_mel(wave)[0, 0].numpy()
    enc = M.TapEncoder(load_teacher(f'{WEIGHTS}/teacher_encoder.safetensors', dtype=torch.bfloat16, device='cuda'))
    ck = torch.load(f'{DATA}/runs/{run_id}/best.pt', map_location='cuda', weights_only=False)
    cfg = P1Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck['config'].items()})
    taps = M.encode_song(enc, mel, sorted(set(cfg.taps) | {20}))
    if pca:
        z = np.load(f'{DATA}/pca/{pca}.npz')
        for l in cfg.taps:
            mu, comp = torch.from_numpy(z[f'mean_{l}']).cuda(), torch.from_numpy(z[f'comp_{l}']).cuda()
            y = (taps[l].cuda().float() - mu) @ comp
            if f'scale_{l}' in z:  # same int8 quantise/dequantise as the training cache
                sc = torch.from_numpy(z[f'scale_{l}']).cuda()
                y = (y / sc).round().clamp(-127, 127) * sc
            taps[l] = y.to('cpu', torch.float16)
    D = int(mel.shape[0] * 10_000)
    song = SongCache({l: taps[l].numpy() for l in cfg.taps}, M.summary_means(taps[20]).numpy(), mel.astype(np.float16), D)
    model = P1Model(cfg).cuda().eval()
    model.load_state_dict(ck['model'])
    ctrl = {'allowed_types': ['tap', 'hold'], **(controls or {})}
    req = {'mode': 'mania_4k', 'observed_events': [], 'regions': [{'region_id': 'r0', 'start_us': 0, 'end_us': D, 'state': 'editable'}],
           'controls': ctrl}
    generated, notes = [], []
    for c0 in range(0, D, F.CORE_US):
        sample = TrainingSample(req, tuple(generated), (), (), {}, build_model_view('mania_4k', (), ()))
        ex = F.build_core(sample, (), c0, D, line_index={})
        holds, runs, heads = core_context(sample, ex, {})
        with torch.autocast('cuda', dtype=torch.bfloat16):
            groups = generate_core(model, ex, song, holds, runs, heads, greedy=greedy, temperature=temperature,
                                   seed=seed * 1000 + c0 // F.CORE_US, device='cuda')
        for g in groups:
            t = c0 + g['t_rel_us']
            for n in g['notes']:
                end = t + n['end_rel_us'] if 'end_rel_us' in n else t
                generated.append(Event(f'g{len(generated)}', t, end, 'hold' if end > t else 'tap', LaneLoc(int(n['lane']))))
                notes.append((t // 1000 - OSU_C_MS, int(n['lane']), (end // 1000 - OSU_C_MS) if end > t else None))
    notes = [(max(t, 0), lane, e) for t, lane, e in notes]
    return {'notes': notes, 'seconds': round(time.time() - t0, 1), 'duration_s': D / 1e6, 'step': ck['step'],
            'notes_per_second': round(len(notes) / max(D / 1e6, 1e-6), 2)}


@app.local_entrypoint()
def main(run_id: str, audio: str, out: str = '', greedy: bool = False, temperature: float = 0.8, seed: int = 0):
    data_bytes = Path(audio).read_bytes()
    r = generate.remote(run_id, data_bytes, greedy=greedy, temperature=temperature, seed=seed)
    name = Path(audio).name
    text = _osu_text(Path(audio).stem, r['notes'], f'BeatFCN {run_id} step{r["step"]}', name)
    dest = Path(out or (Path(audio).with_suffix('').as_posix() + f'.{run_id}.osu'))
    dest.write_text(text, encoding='utf-8')
    print(json.dumps({k: v for k, v in r.items() if k != 'notes'} | {'notes': len(r['notes']), 'out': str(dest)}))
