"""Submit one structure, poll to completion and download available artifacts."""
import argparse
import time
from pathlib import Path
import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=60) as client:
        with args.structure.open("rb") as source:
            response = client.post("/api/jobs", files={"file":(args.structure.name, source)},
                                   data={"model_id":args.model_id} if args.model_id else {})
        response.raise_for_status()
        submitted = response.json()
        print(submitted)
        job_id = submitted["job_id"]
        while True:
            response = client.get(f"/api/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            print(job["status"], job.get("current_step", 0))
            if job["status"] in {"converged", "not_converged", "failed", "quarantined"}:
                break
            time.sleep(2)
        folder = args.out / job_id
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in ("final", "trajectory", "provenance"):
            r = client.get(f"/api/jobs/{job_id}/download/{artifact}")
            if r.status_code == 404:
                continue
            r.raise_for_status()
            filename = job.get("final_file", "final.xyz") if artifact == "final" else {"trajectory":"traj.extxyz", "provenance":"provenance.json"}[artifact]
            (folder / filename).write_bytes(r.content)
        print(f"Artifacts: {folder}")


if __name__ == "__main__":
    main()
