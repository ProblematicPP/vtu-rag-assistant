# Notes folder

Place module notes here using this layout:

```
data/<branch>/<scheme>/sem<N>/<SUBJECT_CODE>/module<N>.<pdf|md|txt>
```

Example:

```
data/cse/2022/sem3/BCS303/module1.pdf
data/cse/2022/sem3/BCS303/module2.pdf
data/cse/2022/sem4/BCS401/module4-dynamic-programming.pdf
```

- `sem3`, `semester3` and `3` are all accepted for the semester folder.
- The file name must start with `module<N>` (`module1.pdf`, `Module_2 notes.pdf`, `mod3.pdf`).
- Several files per module are fine; each becomes its own note.
- Subject names and module titles come from [catalog.yaml](catalog.yaml).

PDFs are git-ignored (they are usually third-party material). The markdown
sample under `cse/2022/sem3/BCS303/` is original demo content so the pipeline
can be tried without any PDFs.

After adding files, either run `docker compose exec api python scripts/ingest.py`,
call `POST /api/v1/notes/sync`, or wait for the Airflow `reindex_vtu_notes` DAG.
