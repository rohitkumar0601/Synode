"""Artifact store over fsspec: local directory or S3-compatible object storage."""

from __future__ import annotations

import io
import json
import os

import fsspec
import pandas as pd
from pydantic import BaseModel


class ArtifactStore:
    def __init__(self, root: str, storage_options: dict | None = None) -> None:
        self.root = root
        self.storage_options = storage_options or {}
        self.fs, base = fsspec.core.url_to_fs(root, **self.storage_options)
        self._base = base.rstrip("/")

    def _path(self, key: str) -> str:
        return f"{self._base}/{key}"

    def exists(self, key: str) -> bool:
        return self.fs.exists(self._path(key))

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        with self.fs.open(path, "wb") as f:
            f.write(data)

    def delete(self, key: str) -> None:
        """Remove one key; a missing key is not an error."""
        path = self._path(key)
        if self.fs.exists(path):
            self.fs.rm(path)

    def modified(self, key: str) -> float:
        """Last-modified time of a key as a POSIX timestamp."""
        return self.fs.modified(self._path(key)).timestamp()

    def read_bytes(self, key: str) -> bytes:
        with self.fs.open(self._path(key), "rb") as f:
            return f.read()

    def write_df(self, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.write_bytes(key, buf.getvalue())

    def read_df(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.read_bytes(key)))

    def write_json(self, key: str, obj: dict | BaseModel) -> None:
        if isinstance(obj, BaseModel):
            text = obj.model_dump_json(indent=2)
        else:
            text = json.dumps(obj, indent=2, default=str)
        self.write_bytes(key, text.encode())

    def read_json(self, key: str) -> dict:
        return json.loads(self.read_bytes(key))

    def list(self, prefix: str) -> list[str]:
        path = self._path(prefix)
        if not self.fs.exists(path):
            return []
        base = self.fs._strip_protocol(self._base).rstrip("/") + "/"
        found = [self.fs._strip_protocol(p) for p in self.fs.find(path)]
        return sorted(p[len(base):] if p.startswith(base) else p for p in found)


def store_from_env() -> ArtifactStore:
    root = os.environ.get("STL_STORAGE_ROOT", "outputs/store")
    options: dict = {}
    if root.startswith("s3://"):
        options = {
            "key": os.environ.get("AWS_ACCESS_KEY_ID"),
            "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "client_kwargs": {"endpoint_url": os.environ.get("S3_ENDPOINT_URL")},
        }
    return ArtifactStore(root, options)
