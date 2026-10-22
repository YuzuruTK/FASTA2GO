"""One tiny progress helper for every slow step.

Two shapes cover everything in this project:

- ``track(items, desc, unit)``: wrap any iterable (proteins, rows, ...).
- ``track_bytes(response, dest, desc)``: stream a ``requests`` response to a
  file with a byte bar driven by ``Content-Length`` (indeterminate when the
  header is missing).

Both are thin wrappers over ``tqdm`` with the project's defaults, so call
sites stay one-liners and the style stays consistent.
"""

from pathlib import Path
from typing import Iterable

from tqdm import tqdm


def track(
    items: Iterable,
    desc: str,
    unit: str = "item",
    **kwargs,
) -> tqdm:
    """Wrap an iterable with a progress bar.

    Extra keyword arguments are passed straight to ``tqdm`` (e.g.
    ``total=``, ``disable=``).
    """
    return tqdm(items, desc=desc, unit=unit, **kwargs)


def track_bytes(
    response,
    dest: str | Path,
    desc: str = "Downloading",
    chunk_size: int = 1 << 20,
) -> int:
    """Stream a requests response to ``dest`` with a byte progress bar.

    Returns the number of bytes written.
    """
    try:
        total = int(response.headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        total = 0
    written = 0
    bar = tqdm(total=total or None, desc=desc, unit="B", unit_scale=True)
    with bar, open(dest, "wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
            bar.update(len(chunk))
    return written
