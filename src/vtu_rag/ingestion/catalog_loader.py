"""Loads `data/catalog.yaml` into typed entries and seeds the database from it."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vtu_rag.db import Database
from vtu_rag.repositories import CatalogRepository

logger = logging.getLogger(__name__)

CATALOG_FILENAME = "catalog.yaml"


@dataclass(frozen=True)
class CatalogSubject:
    code: str
    name: str
    branch: str
    scheme: str
    semester: int
    modules: dict[int, str] = field(default_factory=dict)


def load_catalog(data_dir: Path) -> dict[str, CatalogSubject]:
    """Returns subjects keyed by upper-case subject code. Missing file → empty catalogue."""
    path = data_dir / CATALOG_FILENAME
    if not path.exists():
        logger.warning("No catalogue found at %s", path)
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    subjects: dict[str, CatalogSubject] = {}
    for branch, branch_cfg in (raw.get("branches") or {}).items():
        for scheme, scheme_cfg in (branch_cfg.get("schemes") or {}).items():
            for entry in scheme_cfg.get("subjects") or []:
                code = str(entry["code"]).upper()
                subjects[code] = CatalogSubject(
                    code=code,
                    name=entry["name"],
                    branch=str(branch).lower(),
                    scheme=str(scheme),
                    semester=int(entry["semester"]),
                    modules={int(k): str(v) for k, v in (entry.get("modules") or {}).items()},
                )
    return subjects


async def seed_catalog(db: Database, data_dir: Path) -> int:
    catalog = load_catalog(data_dir)
    async with db.session() as session:
        repo = CatalogRepository(session)
        for item in catalog.values():
            subject = await repo.upsert_subject(
                code=item.code,
                name=item.name,
                branch=item.branch,
                scheme=item.scheme,
                semester=item.semester,
            )
            for number, title in item.modules.items():
                await repo.upsert_module(subject, number, title)
    logger.info("Seeded %d subjects from catalogue", len(catalog))
    return len(catalog)
