"""Shared strict schemas for metadata artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SOURCES = ("category", "brand", "title")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_table(path):
    return pd.read_csv(path, keep_default_na=False)


def flags(values):
    values = np.asarray(values)
    require(all(isinstance(v, (bool, np.bool_)) or v in (0, 1, "True", "False", "0", "1")
                for v in values), "Availability flags must be boolean or 0/1")
    return np.asarray([v in (True, 1, "True", "1") for v in values], dtype=bool)


def categories(value):
    result = json.loads(value)
    require(isinstance(result, list) and all(isinstance(x, str) and x.strip() == x and x
                                          for x in result), "categories must be a JSON list of labels")
    require(len(result) == len(set(result)), "Duplicate category labels")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n")


def check_availability(table):
    require(np.array_equal(table.model_item_id, np.arange(1, len(table) + 1)), "Catalog IDs must be ordered 1..N")
    for source in SOURCES:
        flags(table[f"has_{source}"])


def coverage(table):
    n = len(table)
    return {"total_items": n, **{
        s: {"available": int(flags(table[f"has_{s}"]).sum()),
            "missing": n - int(flags(table[f"has_{s}"]).sum()),
            "coverage_pct": round(float(flags(table[f"has_{s}"]).mean()) * 100, 2) if n else 0.0}
        for s in SOURCES}}
