"""Exercise actual finite-value guards and preservation of usable frames."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from relaxrouter.engine import NumericalFailure, relax
from relaxrouter.telemetry import Telemetry, summarize


def test_nan_after_valid_steps_preserves_history(tmp_path):
    class GoesBad(EMT):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            if self.atoms.positions[0, 0] < 0.049:
                self.results["forces"][:] = np.nan

    atoms = bulk("Cu", cubic=True)
    atoms.positions[0] += [0.05, 0.02, 0]
    frames = []
    with pytest.raises(NumericalFailure):
        relax(atoms, GoesBad(), dict(fmax=1e-9, max_steps=30, relax_cell=False),
              tmp_path, lambda:None, frames.append)
    assert frames and all(np.isfinite(f["energy"]) for f in frames)
    last = read(tmp_path / "traj.extxyz", index=-1)
    assert np.isfinite(last.get_forces()).all()
    assert np.isfinite(last.get_potential_energy())


def test_cell_relaxation(tmp_path):
    atoms = bulk("Cu", a=3.7, cubic=True)
    frames = []
    result = relax(atoms, EMT(), dict(fmax=0.05, max_steps=100, relax_cell=True),
                   tmp_path, lambda:None, frames.append)
    assert result["status"] == "converged"
    assert frames[-1]["convergence_force"] <= 0.05
    assert atoms.cell.lengths()[0] < 3.7


def test_energy_units_and_missing_measurements():
    attempts = [dict(wall_time=2, gpu_energy_j=100, mean_power_w=50, mean_utilization=40),
                dict(wall_time=3, gpu_energy_j=200, mean_power_w=70, mean_utilization=80)]
    result = summarize(attempts, idle_power=10, intensity=0.16)
    assert result["gpu_energy_j"] == 300
    assert result["idle_share"] == pytest.approx(50/300)
    assert result["carbon_kg"] == pytest.approx(300/3_600_000*0.16)
    assert result["mean_utilization"] == 64
    attempts[0]["gpu_energy_j"] = None
    assert summarize(attempts, 10, 0.16)["carbon_kg"] is None
    monitor = Telemetry()
    monitor.prepare("cpu")
    monitor.start()
    assert all(value is None for value in monitor.stop().values())
