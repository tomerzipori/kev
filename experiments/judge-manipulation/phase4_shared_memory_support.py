"""Small JSON and allocator helpers shared by the measured GPU stages."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_stage(path, stage, *, cuda_active=False, **details):
    atomic_json(path, {
        "stage": stage,
        "cuda_active": bool(cuda_active),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    })


class CudaObservation:
    """Sample the allocator in the existing CUDA process, never in the watchdog."""

    def __init__(self, path, stage_name, interval_seconds=1.0):
        self.path = Path(path)
        self.stage_name = stage_name
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="phase4-cuda-observation", daemon=True)
        self.write_lock = threading.Lock()
        self.error = None

    def start(self):
        atomic_json(self.path, {
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "stage": self.stage_name,
            "cuda_active": False,
            "allocator_status": "pre_cuda_import",
            "cuda_current_allocated_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        })
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("CUDA observation thread did not stop")
        if self.error:
            raise RuntimeError(f"CUDA observation failed: {self.error}")

    def _sample(self):
        import torch
        from kev.device import allocated_bytes

        while not self.stop_event.is_set():
            try:
                self.capture(torch, allocated_bytes)
            except BaseException as error:
                self.error = f"{type(error).__name__}: {error}"
                self.stop_event.set()
                return
            self.stop_event.wait(self.interval_seconds)

    def capture(self, torch, allocated_bytes):
        with self.write_lock:
            initialized = torch.cuda.is_initialized()
            values = {
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": time.monotonic_ns(),
                "stage": self.stage_name,
                "cuda_active": initialized,
                "allocator_status": "active" if initialized else "pre_cuda",
                "cuda_current_allocated_bytes": torch.cuda.memory_allocated("cuda") if initialized else None,
                "cuda_peak_allocated_bytes": allocated_bytes("cuda") if initialized else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved("cuda") if initialized else None,
            }
            atomic_json(self.path, values)
            return values

