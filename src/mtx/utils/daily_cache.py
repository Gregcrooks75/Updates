"""Small, single-writer cache helpers shared by the M2T daily stages."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd


def day_label(snapshot_date=None) -> str:
    return date.fromisoformat(str(snapshot_date or date.today())[:10]).isoformat()


def division_name(division: str) -> str:
    value = str(division).strip()
    if not value or value in {".", ".."} or any(c in value for c in '/\\:<>"|?*'):
        raise ValueError(f"Unsafe division directory: {division!r}")
    return value


def frame_signature(frame: pd.DataFrame, keys: list[str]) -> str:
    """Order-independent fingerprint; nulls remain distinct from blank strings."""
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"Fingerprint is missing keys: {sorted(missing)}")
    ordered = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    for column in sorted(ordered.columns):
        digest.update(column.encode("utf-8"))
        digest.update(b"\x00")
        values = ordered[column].astype("string")
        digest.update(pd.util.hash_pandas_object(values, index=False).values.tobytes())
    return digest.hexdigest()


def signature(*items) -> str:
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".w{uuid4().hex[:12]}.tmp"


def atomic_json(value, path: Path) -> None:
    path = Path(path)
    temporary = _temporary(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = _temporary(path)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        check = pd.read_parquet(temporary)
        if len(check) != len(frame) or list(check.columns) != list(frame.columns):
            raise ValueError(f"Parquet verification failed: {path}")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_record(path: Path, root: Path) -> dict:
    return {"path": Path(path).relative_to(root).as_posix(), "sha256": file_digest(path)}


def manifest_valid(manifest, root: Path, expected_signature: str) -> bool:
    if not isinstance(manifest, dict):
        return False
    if manifest.get("complete") is not True or manifest.get("signature") != expected_signature:
        return False
    try:
        files = manifest["files"]
        if not files:
            return False
        for entry in files:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = Path(root) / relative
            if not path.is_file() or file_digest(path) != entry["sha256"]:
                return False
    except (OSError, KeyError, TypeError):
        return False
    return True


def merge_parquet_parts(parts: list[Path], output: Path) -> None:
    """Stream completed batches into the existing one-file-per-division API."""
    import pyarrow.parquet as pq

    if not parts:
        raise ValueError("At least one schema-bearing candidate batch is required.")
    output = Path(output)
    temporary = _temporary(output)
    try:
        expected_rows = 0
        schema = pq.read_schema(parts[0])
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for part in parts:
                table = pq.read_table(part)
                expected_rows += table.num_rows
                writer.write_table(table.cast(schema))
        if pq.read_metadata(temporary).num_rows != expected_rows:
            raise ValueError(f"Incomplete candidate assembly: {output}")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def clean_temporary_files(root: Path) -> None:
    """Remove only our orphaned short filenames; the stages require one writer."""
    for path in Path(root).glob(".w????????????.tmp"):
        path.unlink(missing_ok=True)
