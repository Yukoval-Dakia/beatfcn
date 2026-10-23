"""GPU utilisation sampler (nvidia-smi utilization.gpu, the figure Modal's dashboard shows).

  mon = GpuUtil(); ...; mon.take()  -> {'util_mean': 87.3, 'util_p10': 60.0, 'samples': 120} since last take
"""
from __future__ import annotations
import subprocess
import threading


class GpuUtil:
    def __init__(self, period_ms: int = 500):
        self._vals, self._lock = [], threading.Lock()
        try:
            self._proc = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                                           f'-lms={period_ms}'], stdout=subprocess.PIPE, text=True)
        except OSError:
            self._proc = None
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self._proc.stdout:
            try:
                v = float(line.strip().split(',')[0])
            except ValueError:
                continue
            with self._lock:
                self._vals.append(v)

    def take(self) -> dict:
        with self._lock:
            vals, self._vals = self._vals, []
        if not vals:
            return {'util_mean': None, 'util_p10': None, 'samples': 0}
        s = sorted(vals)
        return {'util_mean': round(sum(vals) / len(vals), 1), 'util_p10': s[len(s) // 10], 'samples': len(vals)}

    def peek(self) -> float | None:
        """Mean utilisation since the last take(), without resetting."""
        with self._lock:
            vals = list(self._vals)
        return round(sum(vals) / len(vals), 1) if vals else None

    def close(self):
        if self._proc:
            self._proc.terminate()
