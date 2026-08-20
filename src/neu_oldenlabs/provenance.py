"""Stamp every analysis output with what produced it.

Without this, a figure and the export it came from can drift apart with nothing
to show which version of the data a delivered report contains.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import RNG_SEED, __version__


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(output: Path, inputs: list[Path], parameters: dict) -> None:
    """Record inputs and parameters for one analysis step.

    A missing input is an error rather than an omitted hash: a provenance file
    that silently lists fewer inputs than the run used is worse than none.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for item in inputs:
        item = Path(item)
        if not item.exists():
            raise FileNotFoundError(f"provenance input does not exist: {item}")
        hashes[str(item.resolve())] = sha256(item)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "seed": RNG_SEED,
        "inputs": hashes,
        "parameters": parameters,
    }
    (output / "provenance.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
