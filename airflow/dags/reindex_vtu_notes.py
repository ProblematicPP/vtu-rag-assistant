"""Re-indexes new or changed VTU notes.

The DAG drives the API rather than importing the application, so the Airflow
image doesn't need the app's dependencies:

    wait_for_api  →  sync_notes  →  report
"""

import logging
import os
from datetime import datetime, timedelta

import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.models.param import Param

API = os.environ.get("VTU_API_BASE_URL", "http://api:8000").rstrip("/")
SCHEDULE = os.environ.get("REINDEX_SCHEDULE", "*/30 * * * *")

log = logging.getLogger(__name__)


@dag(
    dag_id="reindex_vtu_notes",
    description="Index new/changed notes from the data folder into OpenSearch",
    schedule=SCHEDULE,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    params={"force": Param(False, type="boolean", description="Re-index unchanged notes too")},
    tags=["vtu-rag", "ingestion"],
)
def reindex_vtu_notes():
    @task(retries=10, retry_delay=timedelta(seconds=30))
    def wait_for_api() -> dict:
        response = requests.get(f"{API}/health", timeout=15)
        # 503 means Postgres/OpenSearch are down — worth retrying; LLM state doesn't matter here
        if response.status_code != 200:
            raise AirflowException(f"API not ready: HTTP {response.status_code}")
        services = response.json()["services"]
        return {"indexed_chunks": services["opensearch"].get("chunks")}

    @task(execution_timeout=timedelta(hours=1))
    def sync_notes(before: dict, params: dict | None = None) -> dict:
        force = bool((params or {}).get("force", False))
        response = requests.post(
            f"{API}/api/v1/notes/sync", params={"force": str(force).lower()}, timeout=3600
        )
        if response.status_code == 409:
            log.info("Another sync is already running; skipping this run")
            return {"summary": {}, "failed": [], "skipped_run": True, "before": before}
        response.raise_for_status()
        body = response.json()
        failed = [r for r in body["results"] if r["status"] == "failed"]
        return {
            "summary": body["summary"],
            "failed": failed,
            "invalid_paths": body["invalid_paths"],
            "before": before,
        }

    @task
    def report(result: dict) -> None:
        if result.get("skipped_run"):
            return
        log.info("Sync summary: %s", result["summary"])
        for path in result.get("invalid_paths", []):
            log.warning("Invalid note path: %s", path)
        for item in result["failed"]:
            log.error("Failed: %s — %s", item["source_uri"], item["error"])
        if result["failed"]:
            raise AirflowException(f"{len(result['failed'])} note(s) failed to index")

    report(sync_notes(wait_for_api()))


reindex_vtu_notes()
