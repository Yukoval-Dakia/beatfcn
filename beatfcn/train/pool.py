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
import numpy as np
import torch


SLAB_ROWS = {3: 4000, 2: 128}  # size classes by rank: taps [cells, 8, D] per 4000 cells (4 classes up to 16000: evicted slabs
# are reused by most new songs; finer classes kept allocating new slabs and reserved memory grew ~1 GB per 25 steps)


class _Song:
    __slots__ = ('taps', 'summary', 'mel10', 'event', 'nbytes', '_host', 'slabs')

    def __init__(self, taps, summary, mel10, event, nbytes, slabs=()):
        self.taps, self.summary, self.mel10, self.event, self.nbytes = taps, summary, mel10, event, nbytes
        self.slabs = list(slabs)  # (key, slab) device buffers this song's tensors are views of (returned on eviction)


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
        self.resident: OrderedDict[str, _Song] = OrderedDict()  # LRU order: oldest first
        self.pending: dict[str, object] = {}  # sha -> future
        self.pinned: dict[str, int] = {}  # sha -> number of queued batches referencing it: never evicted
        # device memory is recycled in size classes: evicted songs return their slabs, new songs reuse them, so the
        # caching allocator sees no malloc/free churn (varying song sizes fragmented it and stalled the main thread
        # once evictions began: attach time doubled and GPU utilisation fell to ~50%)
        self.free: dict[tuple, list] = {}  # key -> [(slab, event the writer must wait for), ...]
        self.stats = {'loads': 0, 'evictions': 0, 'load_s': 0.0, 'wait_s': 0.0, 'bytes': 0, 'slabs': 0}

    # ------------------------------------------------------------ loading
    def _load(self, sha):
        t0 = time.time()
        try:
            taps_np, summary_np, mel_np = self.load_numpy(sha, mel=self.mel)
        except TypeError:  # loaders without the mel flag (tests)
            taps_np, summary_np, mel_np = self.load_numpy(sha)
        arrays = [np.asarray(taps_np[l]) for l in self.taps] + [np.asarray(summary_np)] + ([np.asarray(mel_np)] if self.mel else [])
        # pageable host -> device copies on the side stream into recycled slabs: no pinned allocation per song
        # (cudaHostAlloc / cudaFreeHost of ~115 MB per load are slow, serialising driver calls) and no device
        # malloc / free per song (see __init__)
        host = [torch.from_numpy(a) for a in arrays]
        dev, slabs = [], []
        if self.cuda:
            with torch.cuda.stream(self.stream):
                for h in host:
                    slab, key = self._slab(h)
                    slab[:h.shape[0]].copy_(h)
                    dev.append(slab[:h.shape[0]])
                    slabs.append((key, slab))
                ev = torch.cuda.Event()
                ev.record(self.stream)
        else:
            dev, ev = host, None
        T = len(self.taps)
        song = _Song({l: t for l, t in zip(self.taps, dev[:T])}, dev[T], dev[T + 1] if self.mel else None, ev,
                     sum(a.nbytes for a in arrays), slabs)
        song._host = None  # pageable copies complete before copy_() returns; nothing to keep alive
        self.stats['loads'] += 1
        self.stats['bytes'] += song.nbytes
        self.stats['load_s'] += time.time() - t0
        return song

    def _slab(self, h):
        """A device buffer for host tensor h [n, ...]: the smallest free slab of h's size class or larger (the
        writer waits for the event recorded when it was freed), else a fresh allocation rounded to the class."""
        rows = SLAB_ROWS.get(h.dim(), 1)
        cls = -(-h.shape[0] // rows) * rows
        base = (tuple(h.shape[1:]), h.dtype)
        with self.lock:
            fits = sorted(k for k in self.free if k[1:] == base and k[0] >= cls and self.free[k])
            if fits:
                key = fits[0]
                slab, ev = self.free[key].pop()
                self.stream.wait_event(ev)  # the main stream's last reads of the previous song must finish first
                return slab, key
        key = (cls, *base)
        self.stats['slabs'] += 1
        return torch.empty((cls, *h.shape[1:]), dtype=h.dtype, device=self.dev), key

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
                song = self.resident.pop(victim)
                if self.cuda and song.slabs:  # recycle: writers wait for the reads already queued on this stream
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream())
                    for key, slab in song.slabs:
                        self.free.setdefault(key, []).append((slab, ev))
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
