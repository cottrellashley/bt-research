"""Sweep worker process: ``python -m nautilus_fi.kernel.worker <snapshot> <k>``.

Reads the memory-mapped data snapshot a session wrote (the panel's ``.npy`` files are shared
with every other worker through the page cache), runs its chunk of strategies on one prepared
context (variants share the expression cache), and writes the results next to the tasks. Run as
its own module, so it never re-imports the caller's script."""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path


def main(snapshot: str, k: int) -> None:
    from nautilus_fi.core.fileio import atomic_write_bytes
    from nautilus_fi.kernel.data import PanelSource, load_panel
    from nautilus_fi.kernel.engine import prepare, simulate

    snap = Path(snapshot)
    meta = pickle.loads((snap / "meta.pkl").read_bytes())
    job = pickle.loads((snap / f"tasks{k}.pkl").read_bytes())
    panel = load_panel(snap / "panel", mmap=True)
    overlays = load_panel(snap / "overlays", mmap=True) if (snap / "overlays").exists() else None
    source = PanelSource(panel, meta["attributes"], meta["series"], meta["constituents"], overlays, meta["fields"], "snapshot", meta["version"])
    strategies = [(i, pickle.loads(p)) for i, p in job["tasks"]]
    prep = prepare([s for _, s in strategies], source, job["start"], job["end"])
    out = {}
    for i, s in strategies:
        r = simulate(s, prep, job["start"], job["end"], job["settings"])
        if not job["keep_weights"]:
            r = dataclasses.replace(r, daily_weights=None)
        r.diagnostics["parallel"] = True
        out[i] = r
    atomic_write_bytes(snap / f"out{k}.pkl", pickle.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
