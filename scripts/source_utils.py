"""Small shared primitives for immutable originals and derived source locators."""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_original(path: Path, data: bytes) -> bool:
    """Create once; an existing original can never be replaced, even with force."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise FileExistsError(f'Original already exists: {path}. Use a new version/snapshot stem.')
        return False
    return True
