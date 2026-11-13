# RelaxRouter

A local atomic-structure relaxation workbench with transparent model routing,
persistent MACE calculators, a single background worker, and retained outcomes.
No frontend build, database server, Docker, Redis or Celery. Runs on Windows,
macOS and Linux; NVIDIA CUDA is optional. The architecture implements the
infrastructure requirements supplied with the project, attributed to Goel, Das &
Goddard, *Infrastructure challenges in scaling AI-driven materials discovery*
(Comput. Mater. Sci., 2027). That future publication has not been independently
verified; the supplied requirements are the implementation specification.

## Install and start

Use Python **3.11**. From the project directory in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m relaxrouter
```

macOS / Linux:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m relaxrouter
```

Open **http://127.0.0.1:8000**. With the environment activated the start command is
`python -m relaxrouter`. Launch one server process, without Uvicorn reload or
multiple workers. Stop with Ctrl+C; an active calculation finishes before shutdown.
The server binds to loopback and is intended for one local user, without authentication.

For an NVIDIA GPU, install a CUDA-enabled PyTorch wheel **before** requirements.
Use the command for your OS and driver from the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/), replacing
`pip` with `.venv\Scripts\python.exe -m pip` on Windows. Do not copy an arbitrary
CUDA version: the correct wheel depends on the installed NVIDIA driver. Confirm:

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Without usable CUDA the app selects CPU automatically. Set
`RELAXROUTER_DEVICE=cpu` to force CPU (PowerShell: `$env:RELAXROUTER_DEVICE='cpu'`).
MACE checkpoints **download on first use**, so the first job needs internet access
and takes longer. Subsequent jobs reuse the calculator, keyed by model and dtype;
different precision requires its own cached model. Checkpoints remain in MACE's
user cache; job artifacts are under `./data/jobs/<id>/`. 3Dmol.js and Chart.js are
loaded from CDNs; without internet, the API and downloads still work but viewer
and charts may be unavailable. SQLite metadata is in `data/jobs.sqlite3`.

## Workflow and routing

Choose a structure, project profile and priority, then preview the route or submit.
Profiles are in `relaxrouter/profiles.json`. Explicit form/API values override
profile defaults. Profile preferred models act as overrides; choose Custom settings
for unrestricted automatic routing. A periodic metal + C + H structure is treated
as a MOF heuristic, not a chemical validation. A molecule has no cell or no PBC.
Element/domain coverage and relative model cost determine the route. Accuracy
selects the highest-cost compatible registry entry; these are heuristics, not
validated uncertainty guarantees. MP registry coverage is declared `all` per the
requested fallback policy; individual checkpoints may reject unsupported species.

Uploads: UTF-8 CIF, POSCAR/CONTCAR/`.vasp`, XYZ and extxyz, up to 5 MiB and 5000
atoms. ASE reads the **last frame** of multi-frame input. Cell relaxation uses
FrechetCellFilter for periodic structures. Final files are CIF for full-rank
periodic structures, XYZ otherwise; the extxyz trajectory preserves cells/PBC.
The bundled metal-organic structure only demonstrates routing, not a validated MOF.

One job includes its full relaxation and any retries/continuations. States are
queued, running, converged, not_converged (step budget), failed (general error),
and quarantined (numerical retries exhausted). Continue restarts FIRE from the
last valid trajectory frame, preserving old frames and metadata; it is available
for non-converged, failed and quarantined jobs. If no valid frame exists it starts
from the input. Fresh step budget = the original configured `max_steps`. A new
optimizer is used; optimizer velocity is not restored. Running jobs interrupted
by process termination become failed on restart; queued jobs are recovered.

Default force threshold: 0.05 eV/Å; step cap: `min(2000, 300 + natoms)`. On cell
relaxations the stopping criterion includes cell forces, while the chart displays
maximum atomic force. Numerical optimizer failures retry in float64 once; generic
load/input failures remain failed. Each invocation has its own retry budget.
Nothing is deleted automatically. Confidence checking compares a second
domain-compatible model, when one exists. It reports absolute energy-per-atom and
maximum vector force differences. No second molecular model is shipped, so that
check is reported unavailable for MACE-OFF jobs. Low confidence is a badge, not a
replacement job status; default energy threshold is 0.05 eV/atom.

## Configuration and optional extras

Environment variables are read at startup:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RELAXROUTER_DATA` | `data` | SQLite, logs and job files directory |
| `RELAXROUTER_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, etc. |
| `RELAXROUTER_AUTO_CONTINUE` | `0` | `1`: one continuation after step cap |
| `RELAXROUTER_RETRY_LIMIT` | `1` | float64 numerical retries per invocation |
| `RELAXROUTER_CARBON_INTENSITY` | `0.16` | kg CO2e/kWh; example: 2024 California average; set your own |
| `RELAXROUTER_CONFIDENCE_THRESHOLD` | `0.05` | eV/atom disagreement threshold |
| `RELAXROUTER_MAX_BATCH_SIZE` | `8` | Reserved limit for optional native batching |

Optional NVML: `python -m pip install nvidia-ml-py` (imports as `pynvml`). For CUDA,
the worker maps the CUDA device UUID to NVML, measures idle power over 10 seconds
after the first model warmup, samples utilization and power at 0.1 s, and differences
cumulative energy counters. Unsupported NVML hardware/APIs return nulls. GPU carbon
= J / 3,600,000 × grid intensity. It excludes CPU, memory and facility overhead.
Idle share uses measured relaxation wall time. Device energy includes other GPU
users; use an otherwise idle device for meaningful measurements. Confidence
single-point calculations and cold starts are excluded from relaxation energy.

Optional `torch-sim` availability is reported in health. **Native batching is not
enabled** in this version: the permitted sequential ASE fallback is used whether
or not torch-sim is installed. Provenance explicitly records `batched=false`,
`batch_size=1` and the reason. No optional package is required for functionality.

Job wall time measures relaxation across attempts; `total_wall_time` additionally
includes cold starts and optional confidence evaluations. Per-step
timing brackets FIRE and evaluation with CUDA synchronization, and reports the
median. Import/load/warmup breakdowns are logged on first use (startup remains
lazy). Reliability rates exclude queued/running jobs. Size bins are [0,300),
[300,600), [600,1000), [1000,infinity). Missing energy is null rather than zero.

## Models and licenses

`relaxrouter/models.py` declares model IDs, loader functions, domains, elements,
relative cost, versions and license notes. To add a model, add a loader accepting
`(device, dtype)` and one `Model` entry to `REGISTRY`. The file includes a commented
example for another package. Install that package separately and ensure it returns
an ASE calculator with forces (and stress if relaxing cells).

Shipped loaders explicitly call `mace_mp(model="small"/"medium")` and
`mace_off(model="small")`, with `device` and `default_dtype`. They are pinned to
`mace-torch==0.3.16`; loader signatures were checked against its installed source.
See [MACE foundation model documentation](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html).
MACE-MP-0 uses MIT model licensing. **MACE-OFF23 uses an Academic Software License**;
review [the model's license](https://github.com/ACEsuit/mace-off) before using it.
Package licenses and model licenses can differ.

## API and examples

Interactive API reference: http://127.0.0.1:8000/docs. API endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/jobs` | Multipart `file` and optional profile, priority, model_id, fmax, max_steps, relax_cell, dtype, confidence_check |
| GET | `/api/jobs` | Newest jobs first |
| GET | `/api/jobs/{id}` | Status, progress, results, router reasons and provenance |
| POST | `/api/jobs/{id}/continue` | Fresh budget on last valid frame |
| GET | `/api/jobs/{id}/download/{final\|trajectory\|provenance}` | Download artifacts |
| POST | `/api/route` | Dry-run multipart file, priority, model_id |
| GET | `/api/models`, `/api/profiles` | Registry/cache and profiles |
| GET | `/api/health` | Device, queue and optional packages |
| GET | `/api/stats`, `/api/report` | Reliability/cost JSON and Markdown report |

```sh
python client.py examples/copper.xyz --model-id mace_mp_small
```

Results download under `results/<job_id>/`. The examples directory also includes
water and a small periodic metal-organic routing demonstration.

## Tests

```sh
python -m pytest -q
```

All tests run on CPU, without pynvml or torch-sim. The real MACE copper test
downloads the small checkpoint on first use. For offline tests only:

```sh
python -m pytest -q -m "not integration"
```

Fast API tests inject ASE EMT to exercise persistence, routing and failure paths
without downloading models; the integration test uses the actual MACE calculator.
No remote is configured and no push is required.

Verified on Windows with Python 3.11.18: **12 tests passed**, including actual
MACE-MP small copper convergence. Also exercised the live API and browser views
for structures, plots and statistics. CUDA/NVML hardware was unavailable for this
verification. Upstream PyTorch/e3nn deprecation and checkpoint-loading warnings
remain; they do not fail the tests.

This workspace also has a ready-to-use Python 3.11 test environment at
`data/venv311`. To run it without a system Python launcher:

```powershell
.\data\venv311\Scripts\python.exe -m pip install -r requirements.txt
.\data\venv311\Scripts\python.exe -m relaxrouter
```
