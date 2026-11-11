import io
import json
import time
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write, read
from fastapi.testclient import TestClient
from relaxrouter.api import create_app
from relaxrouter.config import Config
from relaxrouter.engine import NumericalFailure
from relaxrouter.worker import TERMINAL


@pytest.fixture
def service(tmp_path):
    app = create_app(Config(data=tmp_path, device="cpu"))
    app.state.worker.cache.get = lambda *args: EMT()
    with TestClient(app) as client:
        yield client, app


def submit(client, **fields):
    atoms = bulk("Cu", cubic=True)
    atoms.positions[0] += [0.05, 0.02, 0]
    stream = io.StringIO()
    write(stream, atoms, format="extxyz")
    response = client.post("/api/jobs", files={"file":("copper.xyz", stream.getvalue())}, data=fields)
    assert response.status_code == 201, response.text
    return response.json()["job_id"]


def wait(client, job_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] in TERMINAL:
            # terminal state and atomic provenance are persisted separately
            if job.get("provenance", {}).get("finished") == job.get("finished") and job.get("finished"):
                return job
        time.sleep(0.02)
    pytest.fail("Job did not finish")


def test_api_roundtrip_stats_report(service):
    client, app = service
    job_id = submit(client, model_id="mace_mp_small")
    job = wait(client, job_id)
    assert job["status"] == "converged", job
    assert job["median_step_time"] > 0
    for artifact in ("final", "trajectory", "provenance", "initial"):
        assert client.get(f"/api/jobs/{job_id}/download/{artifact}").status_code == 200
    provenance = client.get(f"/api/jobs/{job_id}/download/provenance").json()
    assert provenance["software"]["ase"]
    assert provenance["batched"] is False and provenance["batch_size"] == 1
    assert job["gpu_energy_j"] is None
    stats = client.get("/api/stats").json()
    assert stats["by_size"]["0–300"]["rates"]["converged"] == 1
    report = client.get("/api/report")
    assert report.status_code == 200 and "synchronization" in report.text
    assert "excludes CPU" in report.text or "exclude CPU" in report.text
    assert client.get("/").status_code == 200


def test_retry_nan_and_quarantine(service, monkeypatch):
    client, app = service
    from relaxrouter import worker
    original = worker.relax
    seen = []
    def flaky(*args, **kwargs):
        seen.append(args[2]["dtype"])
        if len(seen) == 1:
            raise NumericalFailure("NaN forces")
        return original(*args, **kwargs)
    monkeypatch.setattr(worker, "relax", flaky)
    job = wait(client, submit(client))
    assert seen == ["float32", "float64"]
    assert job["status"] == "converged" and len(job["retry_history"]) == 1
    def broken(*args, **kwargs):
        raise NumericalFailure("NaN energy")
    monkeypatch.setattr(worker, "relax", broken)
    job = wait(client, submit(client))
    assert job["status"] == "quarantined" and len(job["attempts"]) == 2


def test_continue_and_fallback(service):
    client, app = service
    job_id = submit(client, max_steps=1, fmax=0.00000001)
    job = wait(client, job_id)
    assert job["status"] == "not_converged"
    n = len(job["history"])
    assert client.post(f"/api/jobs/{job_id}/continue").status_code == 202
    continued = wait(client, job_id)
    assert len(continued["history"]) > n
    assert continued["batch_size"] == 1 and not continued["batched"]
    assert "fallback" in continued["batching_reason"].lower()
    assert len(continued["attempts"]) == 2


def test_validation(service):
    client, _ = service
    assert client.post("/api/jobs", files={"file":("bad.xyz", "bad")}).status_code == 422
    assert client.post("/api/jobs", files={"file":("big.xyz", b"x"*(5*1024*1024+1))}).status_code == 413
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/jobs/missing/download/../../foo").status_code == 404
    assert client.post("/api/jobs", files={"file":("one.xyz", "1\n\nCu 0 0 0\n")}, data={"fmax":"nan"}).status_code == 422


def test_general_failure_is_retained(service):
    client, app = service
    def broken(*args):
        raise RuntimeError("Model unavailable")
    app.state.worker.cache.get = broken
    job_id = submit(client)
    job = wait(client, job_id)
    assert job["status"] == "failed" and "Model unavailable" in job["error"]
    assert client.get(f"/api/jobs/{job_id}/download/provenance").status_code == 200
