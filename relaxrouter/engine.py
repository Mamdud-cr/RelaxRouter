"""ASE FIRE relaxation, synchronized timing, and append-only valid frames."""
import time
from pathlib import Path
from typing import Callable
import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter


class NumericalFailure(RuntimeError):
    """Nonfinite calculation or failed optimizer step; eligible for precision retry."""


def relax(atoms: Atoms, calc, options: dict, directory: Path, sync: Callable,
          record: Callable, start_step: int = 0) -> dict:
    """Run a full fresh FIRE budget; preserve every finite evaluated frame."""
    atoms.calc = calc
    target = FrechetCellFilter(atoms) if options["relax_cell"] and atoms.pbc.any() else atoms
    optimizer = FIRE(target, logfile=str(directory / "optimizer.log"))
    times = []
    steps = 0

    def evaluate(save: bool, step_time: float | None = None) -> tuple[float, float]:
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        effective = target.get_forces()
        if not np.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(effective).all():
            raise NumericalFailure("Nonfinite energy, force or cell stress")
        max_force = float(np.linalg.norm(forces, axis=1).max())
        criterion = float(np.linalg.norm(effective, axis=1).max())
        if save:
            write(directory / "traj.extxyz", atoms, format="extxyz", append=True)
            record(dict(step=start_step + steps, energy=energy, max_force=max_force,
                        convergence_force=criterion, step_time=step_time))
        return energy, criterion

    energy, force = evaluate(True)
    while force > options["fmax"] and steps < options["max_steps"]:
        sync()
        begin = time.perf_counter()
        try:
            optimizer.step()
            energy, force = evaluate(False)
        except Exception as exc:
            raise NumericalFailure(str(exc)) from exc
        finally:
            sync()
        times.append(time.perf_counter() - begin)
        steps += 1
        evaluate(True, times[-1])
    periodic = atoms.pbc.any() and atoms.cell.rank == 3
    final = "final.cif" if periodic else "final.xyz"
    write(directory / final, atoms, format="cif" if periodic else "xyz")
    # extxyz retains partial periodicity and cell information for continuation.
    write(directory / "last.extxyz", atoms, format="extxyz")
    return dict(status="converged" if force <= options["fmax"] else "not_converged",
                steps_taken=steps, energy=energy, final_file=final, step_times=times)
