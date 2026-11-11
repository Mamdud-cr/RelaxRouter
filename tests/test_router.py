from ase import Atoms
from ase.build import bulk, molecule
from relaxrouter.router import route, structure_type


def test_domains_and_priorities():
    assert route(molecule("H2O"))[0] == "mace_off_small"
    assert route(bulk("Cu"))[0] == "mace_mp_medium"
    assert route(bulk("Cu"), "speed")[0] == "mace_mp_small"
    assert route(bulk("Cu"), "accuracy")[0] == "mace_mp_medium"
    framework = Atoms("ZnCH", positions=[[0,0,0],[2,0,0],[3,0,0]], cell=[10]*3, pbc=True)
    assert structure_type(framework) == "mof"
    assert route(framework, "speed")[0] == "mace_mp_small"
    assert route(bulk("Cu") * (11,10,10))[0] == "mace_mp_small"


def test_fallback_and_override():
    chosen, reasons = route(Atoms("Cu", positions=[[0,0,0]]))
    assert chosen == "mace_mp_medium" and any("WARNING" in r for r in reasons)
    assert route(molecule("H2O"), model_id="mace_mp_small")[0] == "mace_mp_small"
