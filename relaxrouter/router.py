"""Transparent domain, element, and cost routing."""
from ase import Atoms
from .models import REGISTRY

# Includes post-transition metals and metalloids commonly present in frameworks.
NONMETALS = set("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn Og".split())


def structure_type(atoms: Atoms) -> str:
    if not atoms.pbc.any() or atoms.cell.rank == 0:
        return "molecule"
    elements = set(atoms.get_chemical_symbols())
    if {"C", "H"} <= elements and elements - NONMETALS:
        return "mof"
    return "inorganic"


def compatible(atoms: Atoms) -> list:
    elements = set(atoms.get_chemical_symbols())
    return [m for m in REGISTRY.values() if structure_type(atoms) in m.domain
            and (m.elements == "all" or elements <= set(m.elements))]


def validate_elements(atoms: Atoms, model_id: str) -> None:
    """Reject unsupported species before a model is queued or downloaded."""
    model = REGISTRY[model_id]
    missing = set(atoms.get_chemical_symbols()) - set(model.elements) if model.elements != "all" else set()
    if missing:
        raise ValueError(f"{model.display_name} does not support element(s): {', '.join(sorted(missing))}")


def route(atoms: Atoms, priority: str = "balanced", model_id: str | None = None) -> tuple[str, list[str]]:
    kind = structure_type(atoms)
    reasons = [f"Detected {kind}; {len(atoms)} atoms."]
    if priority not in ("speed", "balanced", "accuracy"):
        raise ValueError("Priority must be speed, balanced or accuracy")
    if model_id:
        if model_id not in REGISTRY:
            raise ValueError("Unknown model")
        validate_elements(atoms, model_id)
        reasons.append("Manual/profile model override; element coverage checked, domain choice is the user's responsibility.")
        return model_id, reasons
    candidates = compatible(atoms)
    if not candidates:
        validate_elements(atoms, "mace_mp_medium")
        return "mace_mp_medium", reasons + ["WARNING: no compatible domain; falling back to element-compatible MACE-MP medium."]
    costs = {"low": 0, "medium": 1, "high": 2}
    candidates.sort(key=lambda m: costs[m.cost])
    if priority == "speed" or (priority == "balanced" and len(atoms) > 1000):
        chosen = candidates[0]
    elif priority == "balanced":
        chosen = next((m for m in candidates if m.cost == "medium"), candidates[0])
    else:
        chosen = candidates[-1]
    reasons.append(f"{priority} priority selects {chosen.cost} relative cost among {len(candidates)} compatible models.")
    return chosen.id, reasons
