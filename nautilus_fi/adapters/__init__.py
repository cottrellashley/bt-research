"""Per-framework adapters: legacy behaviour as named components over the kernel, kept out of its
core (gap analysis §5).

Each adapter module gives its framework's convention ``PROFILE`` (where one exists), small
components built from kernel primitives, the requirements they cover and the ``PARITY`` cases
(where the framework's golden outputs live, with proposed tolerances). ``ADAPTERS`` lists them
with their status: every adapter is ``unconfirmed`` until its parity cases pass on the
framework's real outputs (``adapters.parity``) and a reviewed change confirms the profile.

=====================  ====================================  ======================
adapter                framework                             profile
=====================  ====================================  ======================
``erin``               fi-backtest-experimentation-erin      ``erin_beta_plus``
``fi_lib_backtesting`` fiq_backtester                        ``fi_lib_backtesting``
``mri``                fi-mdl-macrocredit (MRI v1, v2)       ``MRI_v1``
``fi_lib_research``    fi-lib-research engine                ``fi_lib_research``
``eur_ig_lgbm``        EUR IG LightGBM                       ``EUR_IG``
``composite``          new composite score / xrv             canonical
``assetiq``            assetiq                               canonical
``lta_ucits``          lta-ucits (THFA / TABC)               canonical
``credit_pairs``       credit-pairs                          canonical
``globalrates``        globalrates                           canonical
``systematicpm``       fi-lib-systematicpm                   not captured
=====================  ====================================  ======================
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from nautilus_fi.adapters.parity import (
    ParityCase,
    ParityFailure,
    ParityReport,
    Tolerance,
    compare,
    load_golden,
    profile_confirmation,
    run_parity,
)


@dataclass(frozen=True)
class Adapter:
    name: str
    framework: str
    module: str | None
    order: int | None = None              # the gap analysis's parity order (§8 phase 4)
    note: str = ""

    def load(self):
        if self.module is None:
            raise LookupError(f"{self.name}: {self.note}")
        return importlib.import_module(self.module)

    @property
    def status(self) -> str:
        from nautilus_fi.domain.conventions import PROFILES

        if self.module is None:
            return "not captured"
        prof = getattr(self.load(), "PROFILE", None)
        if prof is not None and prof in PROFILES and PROFILES[prof].confirmed:
            return "confirmed"
        return "unconfirmed"


ADAPTERS: dict[str, Adapter] = {a.name: a for a in (
    Adapter("fi_lib_backtesting", "fiq_backtester", "nautilus_fi.adapters.fi_lib_backtesting", 1),
    Adapter("erin", "fi-backtest-experimentation-erin", "nautilus_fi.adapters.erin", 2),
    Adapter("mri", "fi-mdl-macrocredit (MRI v1 / v2)", "nautilus_fi.adapters.mri", 3),
    Adapter("lta_ucits", "lta-ucits (THFA / TABC)", "nautilus_fi.adapters.lta_ucits", 4),
    Adapter("fi_lib_research", "fi-lib-research", "nautilus_fi.adapters.fi_lib_research", 5),
    Adapter("assetiq", "assetiq", "nautilus_fi.adapters.assetiq", 6),
    Adapter("eur_ig_lgbm", "EUR IG LightGBM", "nautilus_fi.adapters.eur_ig_lgbm"),
    Adapter("composite", "new composite score / xrv", "nautilus_fi.adapters.composite"),
    Adapter("credit_pairs", "credit-pairs", "nautilus_fi.adapters.credit_pairs"),
    Adapter("globalrates", "globalrates", "nautilus_fi.adapters.globalrates"),
    Adapter("systematicpm", "fi-lib-systematicpm", None, note="its requirements were not captured in the gap analysis (the photo ends at the item's first line)"),
)}


def status() -> list[dict]:
    """Every adapter: framework, profile, status, requirements covered and parity cases."""
    out = []
    for a in sorted(ADAPTERS.values(), key=lambda a: (a.order is None, a.order or 0, a.name)):
        if a.module is None:
            out.append({"adapter": a.name, "framework": a.framework, "profile": None, "status": a.status, "requirements": (), "parity": (), "note": a.note})
            continue
        m = a.load()
        out.append({"adapter": a.name, "framework": a.framework, "profile": getattr(m, "PROFILE", None), "status": a.status, "requirements": tuple(getattr(m, "REQUIREMENTS", ())), "parity": tuple(c.name for c in getattr(m, "PARITY", ())), "note": a.note})
    return out


__all__ = ["ADAPTERS", "Adapter", "ParityCase", "ParityFailure", "ParityReport", "Tolerance", "compare", "load_golden", "profile_confirmation", "run_parity", "status"]
