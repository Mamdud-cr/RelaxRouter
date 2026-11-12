"""One persistent thread, durable jobs, and isolated per-job provenance."""
import json
import logging
import queue
import statistics
import threading
import time
from datetime import datetime, timezone
from ase.io import read
from .engine import relax, NumericalFailure
from .models import ModelCache, REGISTRY, versions
from .router import compatible

log = logging.getLogger(__name__)
TERMINAL = {"converged", "not_converged", "failed", "quarantined"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Worker:
    def __init__(self, store, config):
        self.store, self.config = store, config
        self.cache = ModelCache(config.device)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.loop, name="relaxrouter-worker", daemon=True)
        from .telemetry import Telemetry
        self.telemetry = Telemetry()
        self.stopping = threading.Event()

    def start(self) -> None:
        for job in reversed(self.store.all()):
            if job["status"] == "running":
                job.update(status="failed", error="Server stopped during relaxation; valid frames retained.")
                self.store.save(job)
                self.provenance(job)
            elif job["status"] == "queued":
                self.queue.put(job["id"])
        self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        self.queue.put(None)
        self.thread.join()  # let the active scientific calculation finish safely

    def loop(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                if job_id is None:
                    return
                self.run(job_id)
            except Exception:
                log.exception("Worker error for %s", job_id)
            finally:
                self.queue.task_done()

    def provenance(self, job: dict) -> None:
        directory = self.store.directory(job["id"])
        payload = dict(job, software=versions(), model_version=REGISTRY[job["model_id"]].version,
                       device=self.cache.device, gpu_name=self.cache.gpu_name, optimizer="ASE FIRE",
                       cold_starts=self.cache.cold_starts, carbon_intensity=self.config.carbon_intensity,
                       carbon_scope="GPU only; excludes CPU, memory and facility overhead")
        with self.store.lock:
            temporary = directory / "provenance.tmp"
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(directory / "provenance.json")

    def run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        directory = self.store.directory(job_id)
        job.update(status="running", started=now(), batched=False, batch_size=1,
                   batching_reason="Sequential ASE fallback; native torch-sim batching is not enabled.")
        job.pop("finished", None)
        self.store.save(job)
        begin = time.perf_counter()
        attempts = job.setdefault("attempts", [])
        invocation_retries = 0
        continuations = 0
        try:
            while True:
                attempt = dict(started=now(), dtype=job["options"]["dtype"], max_steps=job["options"]["max_steps"])
                attempts.append(attempt)
                source = directory / "traj.extxyz" if (directory / "traj.extxyz").exists() else directory / "initial.extxyz"
                atoms = read(source, index=-1)

                def record(frame: dict) -> None:
                    job.setdefault("history", []).append(frame)
                    job.update(current_step=frame["step"], max_force=frame["max_force"])
                    self.store.save(job)

                attempt_start = None
                try:
                    calc = self.cache.get(job["model_id"], job["options"]["dtype"], atoms)
                    self.telemetry.prepare(self.cache.device)
                    self.cache.synchronize()
                    self.telemetry.start()
                    attempt_start = time.perf_counter()
                    result = relax(atoms, calc, job["options"], directory, self.cache.synchronize,
                                   record, job.get("current_step", 0))
                    result.pop("step_times")
                    job.update(result)
                    attempt["status"] = job["status"]
                except NumericalFailure as exc:
                    attempt.update(status="numerical_failure", error=str(exc))
                    job.setdefault("retry_history", []).append(dict(timestamp=now(), error=str(exc), dtype=job["options"]["dtype"]))
                    if invocation_retries < self.config.retry_limit:
                        invocation_retries += 1
                        job["options"]["dtype"] = "float64"
                        continue
                    job.update(status="quarantined", error=str(exc))
                except Exception as exc:
                    attempt.update(status="failed", error=str(exc))
                    raise
                finally:
                    attempt["finished"] = now()
                    if attempt_start is not None:
                        self.cache.synchronize()
                        attempt.update(wall_time=time.perf_counter()-attempt_start, **self.telemetry.stop())
                    self.store.save(job)
                if job["status"] == "not_converged" and self.config.auto_continue and continuations == 0:
                    continuations += 1
                    job.setdefault("retry_history", []).append(dict(timestamp=now(), reason="automatic continuation"))
                    continue
                break
            if job["status"] in {"converged", "not_converged"} and job["options"]["confidence_check"]:
                self.confidence(job, atoms)
        except Exception as exc:
            log.exception("Job %s failed", job_id)
            job.update(status="failed", error=str(exc))
        finally:
            job.update(finished=now(), wall_time=job.get("wall_time", 0) + time.perf_counter()-begin)
            job["steps_taken"] = max((f["step"] for f in job.get("history", [])), default=0)
            times = [f["step_time"] for f in job.get("history", []) if f.get("step_time") is not None]
            job["step_times"] = times
            job["median_step_time"] = statistics.median(times) if times else None
            from .telemetry import summarize
            job.update(summarize(attempts, self.telemetry.idle_power, self.config.carbon_intensity))
            self.store.save(job)
            self.provenance(job)

    def confidence(self, job: dict, atoms) -> None:
        """Evaluate a different compatible model; no second model means unavailable."""
        import numpy as np
        choices = [m for m in compatible(atoms) if m.id != job["model_id"]]
        if not choices:
            job["confidence"] = dict(available=False, reason="No second compatible model in registry")
            return
        try:
            baseline_e, baseline_f = atoms.get_potential_energy(), atoms.get_forces().copy()
            other = atoms.copy()
            other.calc = self.cache.get(choices[0].id, job["options"]["dtype"], atoms)
            delta = abs(float(other.get_potential_energy() - baseline_e)) / len(atoms)
            force_delta = float(np.linalg.norm(other.get_forces() - baseline_f, axis=1).max())
            job["confidence"] = dict(available=True, model_id=choices[0].id, energy_per_atom_difference=delta,
                                     max_force_difference=force_delta, threshold=self.config.confidence_threshold)
            job["low_confidence"] = delta > self.config.confidence_threshold
        except Exception as exc:
            job["confidence"] = dict(available=False, reason=str(exc))
