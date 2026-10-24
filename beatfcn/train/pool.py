"""GPU-resident song pool: whole-song feature caches (int8 taps, S20 summary, mel10) kept on the device and
sliced there by batch.attach_audio, so DataLoader workers only ship ~50 KB of metadata per core.

Loads run on a background thread (whole-file reads from the volume -> pinned host -> device on a side
stream); the training loop requests the songs of batches it has prefetched and gets them when needed.
Eviction is LRU among songs no queued batch references, bounded by `capacity` songs.
"""
from __future__ import annotations
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import queue
import numpy as np
import torch

STAGE_BYTES = 256 << 20  # pinned staging buffer per loader thread (one tap of a 20-min song is ~250 MB)


class _Song:
    __slots__ = ('taps', 'summary', 'mel10', 'event', 'nbytes', '_host')

    def __init__(self, taps, summary, mel10, event, nbytes):
        self.taps, self.summary, self.mel10, self.event, self.nbytes = taps, summary, mel10, event, nbytes


class SongPool:
    def __init__(self, load_numpy, taps, device='cuda', capacity=160, loaders=4, mel=True):
        """load_numpy(sha) -> (taps {layer: int8/float array [cells,8,D]}, summary [S,8,D], mel10 [frames,128])."""
        self.load_numpy, self.taps, self.dev = load_numpy, tuple(taps), torch.device(device)
        self.capacity = capacity
        self.mel = mel  # keep the 10 ms mel resident (S2 only; the v2.1 model reads none)
        self.cuda = self.dev.type == 'cuda'
        self.stream = torch.cuda.Stream() if self.cuda else None
        self.pool = ThreadPoolExecutor(loaders)
        self.lock = threading.Lock()
        # Host -> device path: each loader thread reads a file straight into a pinned staging buffer allocated once,
        # then copies it to the device asynchronously. Measured on an L40S: with pageable copies in flight, the
        # main thread's attach_audio took 96 ms per micro-batch instead of 2.3 ms (every CUDA call stalls behind a
        # pageable memcpy); per-song pinned allocations (cudaHostAlloc) serialised the same way.
        self.staging = queue.Queue()
        if self.cuda:
            for _ in range(loaders):
                self.staging.put(torch.empty(STAGE_BYTES, dtype=torch.uint8).pin_memory())
        self.resident: OrderedDict[str, _Song] = OrderedDict()  # LRU order: oldest first
        self.pending: dict[str, object] = {}  # sha -> future
        self.pinned: dict[str, int] = {}  # sha -> number of queued batches referencing it: never evicted
        self.stats = {'loads': 0, 'evictions': 0, 'load_s': 0.0, 'wait_s': 0.0, 'bytes': 0}

    # ------------------------------------------------------------ loading
    def _load(self, sha):
        t0 = time.time()
        if self.cuda and getattr(self.load_numpy, 'paths', None):
            dev, nbytes, ev = self._load_staged(sha)
        else:
            try:
                taps_np, summary_np, mel_np = self.load_numpy(sha, mel=self.mel)
            except TypeError:  # loaders without the mel flag (tests)
                taps_np, summary_np, mel_np = self.load_numpy(sha)
            arrays = [np.asarray(taps_np[l]) for l in self.taps] + [np.asarray(summary_np)] + ([np.asarray(mel_np)] if self.mel else [])
            host = [torch.from_numpy(a) for a in arrays]
            nbytes = sum(a.nbytes for a in arrays)
            if self.cuda:  # pinned staging unavailable (loader without .paths): pageable copies on the side stream
                with torch.cuda.stream(self.stream):
                    dev = [h.to(self.dev) for h in host]
                    ev = torch.cuda.Event()
                    ev.record(self.stream)
            else:
                dev, ev = host, None
        T = len(self.taps)
        song = _Song({l: t for l, t in zip(self.taps, dev[:T])}, dev[T], dev[T + 1] if self.mel else None, ev, nbytes)
        song._host = None
        self.stats['loads'] += 1
        self.stats['bytes'] += song.nbytes
        self.stats['load_s'] += time.time() - t0
        return song

    def _load_staged(self, sha):
        """Read each file of the song into this thread's pinned staging buffer (chunked) and DMA it to a fresh
        device tensor on the side stream; returns (device tensors, bytes, event after the last copy)."""
        from .data import npy_header
        stage = self.staging.get()
        try:
            snp = stage.numpy()
            dev, nbytes = [], 0
            copy_ev = torch.cuda.Event()
            for path in self.load_numpy.paths(sha, mel=self.mel):
                with open(path, 'rb', buffering=0) as f:
                    shape, dtype = npy_header(f)
                    out = torch.empty(tuple(shape), dtype=torch.from_numpy(np.empty(0, dtype)).dtype, device=self.dev)
                    flat = out.view(-1).view(torch.uint8)
                    total, off = flat.numel(), 0
                    while off < total:
                        n = min(STAGE_BYTES, total - off)
                        got = 0
                        while got < n:
                            k = f.readinto(snp[got:n])
                            if not k:
                                raise EOFError(f'{path}: short read at {off + got} of {total}')
                            got += k
                        with torch.cuda.stream(self.stream):
                            flat[off:off + n].copy_(stage[:n], non_blocking=True)
                            copy_ev.record(self.stream)
                        copy_ev.synchronize()  # this thread only: the staging buffer is reused next
                        off += n
                    dev.append(out)
                    nbytes += total
            ev = torch.cuda.Event()
            ev.record(self.stream)
            return dev, nbytes, ev
        finally:
            self.staging.put(stage)

    def request(self, shas):
        """Start loading songs that are not resident; mark them pinned (a queued batch needs them)."""
        with self.lock:
            for sha in set(shas):
                self.pinned[sha] = self.pinned.get(sha, 0) + 1
                if sha in self.resident or sha in self.pending:
                    continue
                self.pending[sha] = self.pool.submit(self._load, sha)

    def release(self, shas):
        """A batch has been consumed: its songs may be evicted again (when capacity requires)."""
        with self.lock:
            for sha in set(shas):
                n = self.pinned.get(sha, 0) - 1
                if n > 0:
                    self.pinned[sha] = n
                else:
                    self.pinned.pop(sha, None)

    def get(self, sha):
        """Resident song for attach_audio (blocks until loaded and copied)."""
        with self.lock:
            song = self.resident.get(sha)
            fut = self.pending.get(sha)
        if song is None:
            if fut is None:
                self.request([sha])
                with self.lock:
                    fut = self.pending[sha]
            t0 = time.time()
            song = fut.result()
            self.stats['wait_s'] += time.time() - t0
            with self.lock:
                self.pending.pop(sha, None)
                self.resident[sha] = song
        if self.cuda:
            cur = torch.cuda.current_stream()
            if song.event is not None:  # the current stream waits for the side-stream copies (no host sync)
                cur.wait_event(song.event)
                song.event = None
                song._host = None
            for t in list(song.taps.values()) + [song.summary] + ([song.mel10] if song.mel10 is not None else []):
                t.record_stream(cur)  # the allocator must not reuse this memory (after eviction) before the
                # consumer stream's reads are done
        with self.lock:
            self.resident.move_to_end(sha)
        self._evict()
        return song

    def _evict(self):
        with self.lock:
            while len(self.resident) > self.capacity:
                victim = next((s for s in self.resident if s not in self.pinned), None)
                if victim is None:
                    break
                del self.resident[victim]
                self.stats['evictions'] += 1

    def wait_idle(self):
        """Finish every pending load and its device copies (before a CUDA graph capture: no other thread
        may issue CUDA work while a stream is capturing)."""
        with self.lock:
            futs = list(self.pending.items())
        for sha, fut in futs:
            self.get(sha)
        if self.cuda:
            self.stream.synchronize()

    def resident_bytes(self):
        with self.lock:
            return sum(s.nbytes for s in self.resident.values())
