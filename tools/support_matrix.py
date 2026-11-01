"""Write ``docs/support_matrix.md`` from ``nautilus_fi.support.default_support_matrix()``.

    python tools/support_matrix.py [path]
"""

from __future__ import annotations

import sys
from pathlib import Path

from nautilus_fi.core.registry import SupportStatus
from nautilus_fi.support import default_support_matrix

MEANING = {
    SupportStatus.VALIDATED: "mechanics covered by a fixture or invariant test named in the row",
    SupportStatus.SUPPORTED_WITH_APPROXIMATION: "runs, with the simplification stated in the row",
    SupportStatus.EXPERIMENTAL: "code path exists, not validated end to end",
    SupportStatus.UNSUPPORTED: "refused by the builder",
}


def render() -> str:
    df = default_support_matrix().to_frame()
    lines = ["# Support matrix", "", "Generated from `nautilus_fi.support.default_support_matrix()` by `tools/support_matrix.py`; every run's manifest carries the rows that apply to the products it holds, and `BacktestSettings.allow_support` refuses runs that need anything above the allowed status (default: EXPERIMENTAL allowed, UNSUPPORTED refused).", "", "| Status | Meaning | Count |", "| --- | --- | --- |"]
    lines += [f"| {st.value} | {MEANING[st]} | {int((df['status'] == st.value).sum())} |" for st in MEANING]
    lines += ["", "| Family | Feature | Status | Approximation / notes | Validation case |", "| --- | --- | --- | --- | --- |"]
    for r in df.itertuples(index=False):
        note = "; ".join(x for x in (r.approximation, r.notes) if x)
        lines.append(f"| {r.product_family} | {r.feature} | {r.status} | {note} | {r.validation_case} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "support_matrix.md"
    out.write_text(render())
    print(f"wrote {out}")
