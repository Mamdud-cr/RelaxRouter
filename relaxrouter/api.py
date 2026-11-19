"""Local REST API and static single-page client."""
import io
import json
import logging
import math
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from ase.io import read, write
from .config import Config
from .models import REGISTRY
from .router import route, structure_type
from .storage import Store
from .worker import Worker, now
from .telemetry import available

PROFILES = json.loads(Path(__file__).with_name("profiles.json").read_text())


async def parse_upload(file: UploadFile):
    """Bound upload size before parsing one structure with ASE."""
    content = await file.read(5 * 1024 * 1024 + 1)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 5 MB")
    name = (file.filename or "").lower()
    fmt = "vasp" if name.endswith(("poscar", "contcar", ".vasp")) else {".cif":"cif", ".xyz":"extxyz", ".extxyz":"extxyz"}.get(Path(name).suffix)
    if not fmt:
        raise HTTPException(422, "Use CIF, POSCAR/VASP, XYZ or extxyz")
    try:
        atoms = read(io.StringIO(content.decode("utf-8-sig")), format=fmt, index=-1)
        import numpy as np
        if len(atoms) == 0 or len(atoms) > 5000:
            raise ValueError("Structure must contain 1–5000 atoms")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("Positions and cell must be finite")
        if atoms.pbc.any() and atoms.cell.rank < 3:
            raise ValueError("Periodic structures need a full-rank cell")
        return atoms
    except Exception as exc:
        raise HTTPException(422, f"Could not read structure: {exc}") from exc


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config()
    store = Store(config.data)
    worker = Worker(store, config)

    @asynccontextmanager
    async def lifespan(app):
        logger = logging.getLogger("relaxrouter")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(config.data / "server.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        console = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addHandler(console)
        logger.info("Startup: models lazy; import/load/warmup breakdown logged on first model use")
        worker.start()
        yield
        worker.close()
        logger.removeHandler(handler)
        logger.removeHandler(console)
        handler.close()

    app = FastAPI(title="RelaxRouter", lifespan=lifespan)
    app.state.worker, app.state.store = worker, store

    def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/profiles")
    def profiles():
        return PROFILES

    @app.get("/api/models")
    def models():
        return [dict(id=m.id, display_name=m.display_name, domain=m.domain, elements=m.elements,
                     relative_cost=m.cost, license_note=m.license_note, version=m.version,
                     loaded=any(k[0] == m.id for k in worker.cache.cache)) for m in REGISTRY.values()]

    @app.get("/api/health")
    def health():
        return dict(device=worker.cache.device, gpu_name=worker.cache.gpu_name, queue_length=worker.queue.qsize(),
                    pynvml=available("pynvml"), batching_enabled=False,
                    device_initialized=bool(worker.cache.cache), requested_device=config.device)

    @app.post("/api/route")
    async def dry_route(file: UploadFile = File(...), priority: str = Form("balanced"), model_id: str | None = Form(None)):
        atoms = await parse_upload(file)
        try:
            chosen, reasons = route(atoms, priority, model_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return dict(model_id=chosen, reasons=reasons, natoms=len(atoms), structure_type=structure_type(atoms))

    @app.post("/api/jobs", status_code=201)
    async def submit(file: UploadFile = File(...), profile: str | None = Form(None), priority: str | None = Form(None),
                     model_id: str | None = Form(None), fmax: float | None = Form(None), max_steps: int | None = Form(None),
                     relax_cell: bool | None = Form(None), dtype: str = Form("float32"), confidence_check: bool = Form(False)):
        atoms = await parse_upload(file)
        selected = next((p for p in PROFILES if p["name"] == profile), None)
        if profile and selected is None:
            raise HTTPException(422, "Unknown profile")
        selected = selected or {}
        priority = priority or selected.get("priority", "balanced")
        fmax = fmax if fmax is not None else selected.get("fmax", 0.05)
        max_steps = max_steps if max_steps is not None else min(2000, 300 + len(atoms))
        if not math.isfinite(fmax) or fmax <= 0 or max_steps <= 0 or dtype not in ("float32", "float64"):
            raise HTTPException(422, "Use a positive finite fmax, positive max_steps, and float32 or float64")
        try:
            chosen, reasons = route(atoms, priority, model_id or selected.get("preferred_model"))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        job_id = uuid.uuid4().hex
        directory = store.directory(job_id)
        directory.mkdir()
        write(directory / "initial.extxyz", atoms, format="extxyz")
        write(directory / "initial.xyz", atoms, format="xyz")
        job = dict(id=job_id, created=now(), status="queued", filename=Path(file.filename or "structure").name,
                   natoms=len(atoms), structure_type=structure_type(atoms), model_id=chosen, router_reasons=reasons,
                   profile=profile, history=[], retry_history=[], attempts=[],
                   options=dict(priority=priority, fmax=fmax, max_steps=max_steps,
                                relax_cell=relax_cell if relax_cell is not None else selected.get("relax_cell", False),
                                dtype=dtype, confidence_check=confidence_check))
        store.save(job)
        worker.provenance(job)
        worker.queue.put(job_id)
        return dict(job_id=job_id, model_id=chosen, reasons=reasons)

    @app.get("/api/jobs")
    def list_jobs():
        return [{k:v for k,v in j.items() if k not in ("history", "step_times")} for j in store.all()]

    @app.get("/api/jobs/{job_id}")
    def detail(job_id: str):
        job = get_job(job_id)
        path = store.directory(job_id) / "provenance.json"
        with store.lock:
            job["provenance"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return job

    @app.get("/api/jobs/{job_id}/download/{kind}")
    def download(job_id: str, kind: str):
        job = get_job(job_id)
        filename = {"initial":"initial.xyz", "final":job.get("final_file"), "trajectory":"traj.extxyz", "provenance":"provenance.json"}.get(kind)
        if not filename or not (store.directory(job_id) / filename).exists():
            raise HTTPException(404, "Artifact not available")
        return FileResponse(store.directory(job_id) / filename, filename=filename)

    @app.post("/api/jobs/{job_id}/continue", status_code=202)
    def continuation(job_id: str):
        with store.lock:
            job = get_job(job_id)
            if job["status"] not in ("not_converged", "failed", "quarantined"):
                raise HTTPException(409, "Only stopped, non-converged jobs can continue")
            job.update(status="queued", continued=now())
            job.pop("error", None)
            store.save(job)
            worker.queue.put(job_id)
        return dict(job_id=job_id, status="queued")

    @app.get("/api/stats")
    def stats():
        from .stats import statistics_for
        return statistics_for(store.all())

    @app.get("/api/report", response_class=PlainTextResponse)
    def report():
        from .stats import report_for
        return PlainTextResponse(report_for(store.all(), worker, config), headers={"Content-Disposition":"attachment; filename=relaxrouter-report.md"})

    return app


app = create_app()
