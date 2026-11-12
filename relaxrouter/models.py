"""Extensible registry and process-lifetime calculator cache."""
import importlib.metadata
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


def mp_small(device: str, dtype: str) -> Any:
    from mace.calculators import mace_mp
    return mace_mp(model="small", device=device, default_dtype=dtype)


def mp_medium(device: str, dtype: str) -> Any:
    from mace.calculators import mace_mp
    return mace_mp(model="medium", device=device, default_dtype=dtype)


def off_small(device: str, dtype: str) -> Any:
    from mace.calculators import mace_off
    return mace_off(model="small", device=device, default_dtype=dtype)


@dataclass(frozen=True)
class Model:
    id: str
    display_name: str
    loader: Callable
    domain: tuple[str, ...]
    elements: tuple[str, ...] | str
    cost: str
    license_note: str
    version: str


REGISTRY = {
    m.id: m for m in [
        Model("mace_mp_small", "MACE-MP small", mp_small, ("inorganic", "mof"), "all", "low", "MACE-MP: MIT model license", "MP-0 small 2023-12-10"),
        Model("mace_mp_medium", "MACE-MP medium", mp_medium, ("inorganic", "mof"), "all", "medium", "MACE-MP: MIT model license", "MP-0 medium 2023-12-03"),
        Model("mace_off_small", "MACE-OFF small", off_small, ("molecule",), tuple("H C N O F P S Cl Br I".split()), "low", "Academic Software License; review before use", "OFF23 small"),
    ]
}
# To add another package: define def other(device, dtype): return OtherCalculator(...)
# then REGISTRY['other'] = Model('other', 'Other', other, ('inorganic',), 'all',
#                              'high', 'Review package/model license', 'checkpoint-v1').


def versions() -> dict:
    """Read installed versions without importing large libraries."""
    result = {}
    for name in ("mace-torch", "torch", "ase"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


class ModelCache:
    """Only the single science worker accesses calculators, keyed by precision."""
    def __init__(self, device: str = "auto"):
        self.requested_device = device
        self.device = "cpu"
        self.gpu_name = None
        self.cache: dict[tuple, Any] = {}
        self.cold_starts: list[dict] = []

    def get(self, model_id: str, dtype: str, atoms: Any) -> Any:
        key = (model_id, dtype)
        if key in self.cache:
            return self.cache[key]
        start = time.perf_counter()
        import torch
        import mace.calculators  # noqa: F401
        imported = time.perf_counter()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.requested_device == "auto" else self.requested_device
        if self.device.startswith("cuda"):
            self.gpu_name = torch.cuda.get_device_name(self.device)
        calc = REGISTRY[model_id].loader(self.device, dtype)
        self.synchronize()
        loaded = time.perf_counter()
        warm = atoms.copy()
        warm.calc = calc
        import numpy as np
        from .engine import NumericalFailure
        if not np.isfinite(warm.get_potential_energy()) or not np.isfinite(warm.get_forces()).all():
            raise NumericalFailure("Nonfinite energy or forces during warmup")
        self.synchronize()
        self.cache[key] = calc
        record = dict(model=model_id, dtype=dtype, library_import_s=imported-start,
                      model_load_s=loaded-imported, warmup_s=time.perf_counter()-loaded)
        self.cold_starts.append(record)
        log.info("Cold start breakdown: %s", record)
        return calc

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)
