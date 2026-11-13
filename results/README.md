# Published result summaries

[`paper/`](paper/) contains the verified summary bundle for the official F2 study.
Run `python scripts/paper_results.py verify` from the checkout root.

CSV metrics use the original 0–1 scale unless the column/caption explicitly says
otherwise. This summary bundle is separate from the full execution archive and
does not include weights, raw reviews, item metadata, or per-user predictions.

Historical local result folders are retained on disk and ignored by Git.
