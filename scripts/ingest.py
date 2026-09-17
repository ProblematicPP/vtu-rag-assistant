"""Index notes from the data folder.

Usage (inside the api container):
    python scripts/ingest.py                 # index new/changed notes
    python scripts/ingest.py --force         # re-index everything
    python scripts/ingest.py --path cse/2022/sem3/BCS303/module1.pdf
"""

import argparse
import asyncio
import json
import sys

from vtu_rag.config import get_settings
from vtu_rag.container import Container
from vtu_rag.ingestion.path_parser import NotePathError, parse_note_path
from vtu_rag.ingestion.sources import DiscoveredNote, LocalFolderSource
from vtu_rag.logging_config import configure_logging
from vtu_rag.models import SourceType


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = Container.build(settings)
    try:
        await container.startup()
        if args.path:
            path = (settings.data_dir / args.path).resolve()
            try:
                location = parse_note_path(path, settings.data_dir)
            except NotePathError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            note = DiscoveredNote(
                location=location,
                source_uri=location.relative_path,
                source_type=SourceType.UPLOAD,
                extension=path.suffix.lower(),
                content=path.read_bytes(),
                title=location.title,
            )
            outcome = await container.ingestion.ingest(note, force=args.force)
            print(json.dumps(outcome.__dict__, default=str, indent=2))
            return 0 if outcome.status != "failed" else 1

        report = await container.ingestion.sync(
            LocalFolderSource(settings.data_dir), force=args.force
        )
        for outcome in report.outcomes:
            detail = f"{outcome.chunks} chunks" if outcome.status == "indexed" else ""
            print(f"[{outcome.status:>7}] {outcome.source_uri} {detail} {outcome.error or ''}")
        for invalid in report.invalid_paths:
            print(f"[invalid] {invalid}")
        print(json.dumps(report.summary()))
        return 1 if report.count("failed") else 0
    finally:
        await container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="re-index even if unchanged")
    parser.add_argument("--path", help="single note path, relative to DATA_DIR")
    sys.exit(asyncio.run(main(parser.parse_args())))
