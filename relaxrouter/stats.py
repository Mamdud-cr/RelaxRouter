"""Reliability denominators and GPU-only cost reporting."""
import platform
import statistics
from .models import versions
from .worker import TERMINAL


def size_bin(n: int) -> str:
    """Nonoverlapping bins: [0,300), [300,600), [600,1000), [1000,+inf)."""
    return "0–300" if n < 300 else "300–600" if n < 600 else "600–1000" if n < 1000 else "1000+"


def rates(jobs: list) -> dict:
    completed = [j for j in jobs if j["status"] in TERMINAL]
    return dict(total=len(jobs), terminal=len(completed),
                counts={s:sum(j["status"] == s for j in completed) for s in sorted(TERMINAL)},
                rates={s:sum(j["status"] == s for j in completed)/len(completed) if completed else None for s in sorted(TERMINAL)})


def statistics_for(jobs: list) -> dict:
    models = {}
    for name in sorted({j["model_id"] for j in jobs}):
        group = [j for j in jobs if j["model_id"] == name and j["status"] in TERMINAL]
        def median(key):
            values = [j[key] for j in group if j.get(key) is not None]
            return statistics.median(values) if values else None
        measured = [j["gpu_energy_j"] for j in group if j.get("gpu_energy_j") is not None]
        models[name] = dict(jobs=len(group), median_step_time=median("median_step_time"),
                            median_relaxation_time=median("wall_time"), total_gpu_energy_j=sum(measured) if measured else None)
    measured = [j for j in jobs if j.get("gpu_energy_j") is not None]
    total = sum(j["gpu_energy_j"] for j in measured)
    unsuccessful = sum(j["gpu_energy_j"] for j in measured if j["status"] in {"not_converged", "failed", "quarantined"})
    return dict(overall=rates(jobs), by_size={b:rates([j for j in jobs if size_bin(j["natoms"]) == b]) for b in ("0–300", "300–600", "600–1000", "1000+")},
                by_type={t:rates([j for j in jobs if j["structure_type"] == t]) for t in ("inorganic", "mof", "molecule")},
                by_model=models, total_gpu_energy_j=total if measured else None,
                unsuccessful_gpu_energy_share=unsuccessful/total if total else None,
                energy_measured_jobs=len(measured), bin_definition="[0,300), [300,600), [600,1000), [1000,infinity)")


def report_for(jobs: list, worker, config) -> str:
    import json
    stats = statistics_for(jobs)
    return f"""# RelaxRouter infrastructure report

## Hardware and software
OS: {platform.platform()}; CPU: {platform.processor() or platform.machine()}.
Python: {platform.python_version()}; device: {worker.cache.device}; GPU: {worker.cache.gpu_name or 'not initialized / CPU'}.
Software: {json.dumps(versions())}.

## Workload and reliability
Submitted structures: {len(jobs)}. Size bins: {stats['bin_definition']}.
Rates use terminal jobs as denominator; queued/running jobs are excluded.
Each job represents one relaxation, including its continuations and retries. All outcomes are retained.
Counts and convergence, non-convergence, failure and quarantine rates:

```json
{json.dumps(stats, indent=2)}
```

## Cost and timing
Per-step times include FIRE step plus energy/force evaluation, bounded by CUDA
synchronization before and after each timed call. Job median uses completed steps.
Wall time measures relaxation only, summed across attempts. Total wall time also
includes loading, warmup, idle calibration and confidence checks, and is retained
separately in provenance. Failed attempts retain finite frames and step times.
Models persist in one thread/process; library import, checkpoint load and warmup
times are logged separately on each first model/precision use.

## GPU utilization, energy and carbon
Optional NVML cumulative energy counter difference in joules; utilization and
power sampled every 0.1 s. Unsupported measurements are null, never zero-filled.
Idle power is measured for 10 s after the first CUDA model warmup.
Idle share = idle power × measured relaxation wall time / energy.
Grid intensity: {config.carbon_intensity} kg CO2e/kWh (default example: 2024
California average; set your own). Carbon = GPU joules / 3,600,000 × intensity.
GPU-only estimates exclude CPU, memory and facility overhead. Confidence evaluations
are outside relaxation energy measurements. Energy may include other device users.

## Retry policy and budgets
Default max_steps = min(2000, 300 + natoms); positive user overrides accepted.
FIRE default fmax = 0.05 eV/Å. Numerical failures retry in float64 up to
{config.retry_limit} times per invocation; exhaustion is quarantined. Other
exceptions are failed. One automatic continuation on step cap: {config.auto_continue}.
Manual continuation starts FIRE from the last valid frame with a fresh step budget.
All jobs use sequential ASE with batch_size=1; native batching is not implemented.
Per-job provenance contains exact options, model versions, attempts and router reasons.
"""
