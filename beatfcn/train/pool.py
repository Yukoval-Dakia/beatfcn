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


class _Song:
    __slots__ = ('taps', 'summary', 'mel10', 'event', 'nbytes', '_host')

    def __init__(self, taps, summary, mel10, event, nbytes):
        self.taps, self.summary, self.mel10, self.event, self.nbytes = taps, summary, mel10, event, nbytes


class SongPool:
    def __init__(self, load_numpy, taps, device='cuda', capacity=160, loaders=4):
        """load_numpy(sha) -> (taps {layer: int8/float array [cells,8,D]}, summary [S,8,D], mel10 [frames,128])."""
        self.load_numpy, self.taps, self.dev = load_numpy, tuple(taps), torch.device(device)
        self.capacity = capacity
        self.cuda = self.dev.type == 'cuda'
        self.stream = torch.cuda.Stream() if self.cuda else None
        self.pool = ThreadPoolExecutor(loaders)
        self.lock = threading.Lock()
        self.resident: OrderedDict[str, _Song] = OrderedDict()  # LRU order: oldest first
        self.pending: dict[str, object] = {}  # sha -> future
        self.pinned: dict[str, int] = {}  # sha -> number of queued batches referencing it: never evicted
        self.stats = {'loads': 0, 'evictions': 0, 'load_s': 0.0, 'wait_s': 0.0, 'bytes': 0}

    # ------------------------------------------------------------ loading
    def _load(self, sha):
        t0 = time.time()
        taps_np, summary_np, mel_np = self.load_numpy(sha)
        arrays = [np.asarray(taps_np[l]) for l in self.taps] + [np.asarray(summary_np), np.asarray(mel_np)]
        host = [torch.from_numpy(a).pin_memory() if self.cuda else torch.from_numpy(a) for a in arrays]
        if self.cuda:
            with torch.cuda.stream(self.stream):
                dev = [h.to(self.dev, non_blocking=True) for h in host]
                ev = torch.cuda.Event()
                ev.record(self.stream)
        else:
            dev, ev = host, None
        song = _Song({l: t for l, t in zip(self.taps, dev[:-2])}, dev[-2], dev[-1], ev, sum(a.nbytes for a in arrays))
        song._host = host if self.cuda else None  # keep pinned buffers alive until the copies complete
        self.stats['loads'] += 1
        self.stats['bytes'] += song.nbytes
        self.stats['load_s'] += time.time() - t0
        return song

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
            for t in list(song.taps.values()) + [song.summary, song.mel10]:  # the allocator must not reuse this
                t.record_stream(cur)  # memory (after eviction) before the consumer stream's reads are done
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
