# RelaxRouter

RelaxRouter is a local web app for relaxing atomic structures with pretrained MACE potentials on CPU or an optional NVIDIA GPU. FastAPI, SQLite and a build-free UI handle model selection and ASE relaxation.

## Quick start

Use Python **3.11** from the project directory.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m relaxrouter
```

**macOS / Linux (untested)**

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m relaxrouter
```

Open **http://127.0.0.1:8000**.

Use one server process on single-user loopback, without authentication.

Models download on first use; checkpoints are cached on disk across restarts without re-downloading, while calculators stay in memory only while the server runs.

## GPU setup

Install the CUDA PyTorch wheel from the official [selector](https://pytorch.org/get-started/locally/) first, inside the virtual environment.
Run `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
Recheck after installing requirements: pip can replace the CUDA wheel; otherwise the app falls back to CPU.

## Usage

Choose a file and profile, preview the route, then submit. Profile names match `relaxrouter/profiles.json`.

| Profile | Priority | Preferred model | Relax cell |
| --- | --- | --- | --- |
| MOF screening | Speed | Automatic | Yes |
| bulk crystals | Balanced | Automatic | Yes |
| molecules | Speed | MACE-OFF small | No |

Non-periodic structures are molecules; periodic metal-plus-C-and-H structures are classified as MOFs (a heuristic), otherwise inorganic. Element/domain filtering precedes priority selection: Speed chooses lowest cost; Balanced prefers medium, or low above 1000 atoms; Accuracy chooses highest available cost. Preferred models bypass automatic selection; use Custom settings with Automatic routing to avoid that override.

Supported element sets are in `relaxrouter/models.py` and `/api/models`; unsupported elements are rejected before queueing, including manual overrides.

Inputs: CIF, POSCAR/CONTCAR/VASP, XYZ or extxyz; limits **5 MiB**, **5000 atoms**; default `fmax=0.05` eV/Å and step cap `min(2000, 300 + natoms)`.

States: `queued`, `running`, `converged`, `not_converged`, `failed`, `quarantined`.
Numerical failures retry once in float64 by default, then become quarantined; other exceptions become failed.
Continue restarts non-converged, failed or quarantined jobs from the last valid frame (or input), with a fresh original step budget.

Optional model comparison flags energy differences above 0.05 eV/atom; MACE-MP small/medium share training data, so agreement is weak evidence of reliability.

Job files, `provenance.json` and the SQLite database are stored under `data/`.
To add a model, add a loader and a Model entry in relaxrouter/models.py (see the commented example).

## Paper motivation and implementation

The primary infrastructure reference is **Naveen Goel, Tridip Das and William A.
Goddard III, [*Infrastructure challenges in scaling AI-driven materials discovery:
a critical review*](https://doi.org/10.1016/j.commatsci.2026.115212), Computational
Materials Science 276, 115212 (2027)** [1]. RelaxRouter applies selected recommendations:

| Idea from the review | Implementation |
| --- | --- |
| Reduce startup overhead (§3.3). | A persistent worker reuses models and runs one full relaxation per job. |
| Retain difficult candidates (§§3.3–3.4). | Failures remain available for retries/continuation; size-aware budgets and outcome statistics expose selection effects. |
| Preserve reproducibility (§3.6; §4.4). | Trajectories, settings, versions and retry histories accompany each result and infrastructure report. |
| Report computational cost (§§3.3, 4.5). | Synchronized timing and optional NVML provide utilization, energy and GPU-only carbon estimates. |

This implements selected principles, not the paper's 120-MOF benchmark or full
reporting checklist. Routing, the step-budget formula and model comparison are
project heuristics. Distributed execution and native batching are not implemented.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `RELAXROUTER_DATA` | `data` | SQLite, logs and job files directory |
| `RELAXROUTER_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, etc. |
| `RELAXROUTER_AUTO_CONTINUE` | `0` | `1`: one continuation after step cap |
| `RELAXROUTER_RETRY_LIMIT` | `1` | float64 numerical retries per invocation |
| `RELAXROUTER_CARBON_INTENSITY` | `0.16` | kg CO2e/kWh; example: 2024 California average; set your own |
| `RELAXROUTER_CONFIDENCE_THRESHOLD` | `0.05` | eV/atom disagreement threshold |

Optional GPU telemetry: `python -m pip install nvidia-ml-py` (imports as `pynvml`).

## API

API reference: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

```sh
python client.py examples/copper.xyz --model-id mace_mp_small
```

## Tests

```sh
python -m pytest -q
python -m pytest -q -m "not integration"
```

**Tested on:** Windows, Python 3.11; CUDA/NVML hardware not verified.

## License and citation

RelaxRouter code/documentation use the MIT License; see [LICENSE](LICENSE). Dependencies and weights retain their licenses: MACE-MP-0 is MIT; MACE-OFF23 has an Academic Software License. Cite the review [1], model used [2–4], ASE [5] and FIRE [6].

## References

1. N. Goel, T. Das and W. A. Goddard III.
   *Infrastructure challenges in scaling AI-driven materials discovery: a critical
   review.* Computational Materials Science **276**, 115212 (2027).
   [doi:10.1016/j.commatsci.2026.115212](https://doi.org/10.1016/j.commatsci.2026.115212).
2. I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner and
   G. Csányi. *MACE: Higher Order Equivariant Message Passing Neural Networks for
   Fast and Accurate Force Fields.* Advances in Neural Information Processing
   Systems **35**, 11423–11436 (2022).
   [arXiv:2206.07697](https://arxiv.org/abs/2206.07697).
3. I. Batatia et al. *A foundation model for atomistic materials
   chemistry.* arXiv preprint, first posted 2023.
   [arXiv:2401.00096](https://arxiv.org/abs/2401.00096).
4. D. P. Kovács et al. *MACE-OFF: Transferable Short Range Machine
   Learning Force Fields for Organic Molecules.* arXiv preprint, first posted 2023.
   [arXiv:2312.15211](https://arxiv.org/abs/2312.15211).
5. A. Hjorth Larsen et al. *The atomic simulation environment—a Python
   library for working with atoms.* Journal of Physics: Condensed Matter **29**,
   273002 (2017).
   [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).
6. E. Bitzek, P. Koskinen, F. Gähler, M. Moseler and P. Gumbsch.
   *Structural Relaxation Made Simple.* Physical Review Letters **97**, 170201
   (2006). [doi:10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201).
