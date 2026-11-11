"""Real CPU smoke test; requires first-use model download, no GPU or extras."""
import pytest
from ase.build import bulk
from relaxrouter.engine import relax
from relaxrouter.models import ModelCache


@pytest.mark.integration
def test_perturbed_copper_mace(tmp_path):
    import torch
    torch.set_num_threads(2)
    atoms = bulk("Cu", cubic=True)
    atoms.positions[0] += [0.04, 0.03, 0.01]
    cache = ModelCache("cpu")
    calc = cache.get("mace_mp_small", "float32", atoms)
    assert cache.get("mace_mp_small", "float32", atoms) is calc
    frames = []
    result = relax(atoms, calc, dict(fmax=0.05, max_steps=100, relax_cell=False), tmp_path, cache.synchronize, frames.append)
    assert result["status"] == "converged"
    assert frames[-1]["max_force"] <= 0.05
    assert len(cache.cold_starts) == 1
