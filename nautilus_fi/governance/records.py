"""Official backtest records: levels, gates and an append-only, hash-chained registry.

A result climbs EXPLORATORY -> CANDIDATE -> OFFICIAL only through gates that check what the
research standards require. From CANDIDATE up:

* a hypothesis registered (registry-stamped, content frozen by hash) before its first trial, the
  trial in the ledger, and machine-checkable failure criteria that the result does not meet;
* conventions confirmed by their values (relabelling a legacy profile does not confirm it);
* no filled or altered data (zero fills, refills, unexplained exits, benchmark / cash / cost
  fills, dropped dates), no hard-constraint violations or relaxed constructions, look-ahead
  checks on, and at least ``min_years`` of track record;
* the deflated Sharpe ratio and PBO over the whole search the result was selected from (every
  trial on the same universe, overlapping window and data version, whatever hypothesis it was filed
  under, or none), at least ``min_dsr`` / at most ``max_pbo``;
* the holdout untouched by any trial (for OFFICIAL: used only by the promoted spec).

OFFICIAL also needs an Implement run of the same strategy (fingerprint) that completed solvent with
manifest hashes, a reconciliation with its content and within tolerance, and point-in-time inputs.
One record per result and level. A failed gate raises ``GateFailure`` listing every failed check;
nothing is written for a refused promotion. Submissions hold a lock while they read the chain head
and append, and keep the head's hash in a separate anchor file, so a deleted or forked tail fails
``verify_chain``. The thresholds' defaults are placeholders until the team sets them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from nautilus_fi.domain.conventions import Annualisation, unconfirmed_profile


class RecordLevel(str, Enum):
    EXPLORATORY = "EXPLORATORY"
    CANDIDATE = "CANDIDATE"
    OFFICIAL = "OFFICIAL"


class GateFailure(RuntimeError):
    def __init__(self, level: RecordLevel, failed: dict[str, str]) -> None:
        self.level, self.failed = level, failed
        super().__init__(f"{level.value} gate failed: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))


@dataclass
class OfficialRecord:
    record_id: str
    level: str
    name: str
    hypothesis_id: str | None
    spec_hash: str
    conventions: dict
    paper: dict
    implement: dict | None
    reconciliation: dict | None
    dsr: dict | None
    pbo: dict | None
    n_trials: int
    checks: dict
    data_hashes: dict = field(default_factory=dict)
    code_version: str | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    prev_hash: str = ""
    result_key: str = ""
    thresholds: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)            # the operational store's ids (LINK_KEYS): portfolio, PORT id, config version, owner
    config: str | None = None                            # the strategy as code (Strategy.to_code), when it can be saved that way
    version: int = 1                                     # records of this exact name so far, this one included
    hash: str = ""

    def to_markdown(self) -> str:
        lines = [f"# {self.level} record {self.record_id}: {self.name}", "", f"hypothesis `{self.hypothesis_id}` · spec `{self.spec_hash}` · created {self.created_utc}", "", "## Checks", "", "| check | ok | detail |", "| --- | --- | --- |"]
        lines += [f"| {k} | {'yes' if v['ok'] else 'NO'} | {v['detail']} |" for k, v in self.checks.items()]
        lines += ["", "## Paper", "", *(f"* {k}: {v}" for k, v in self.paper.items() if not isinstance(v, dict))]
        if self.implement:
            lines += ["", "## Implement", "", *(f"* {k}: {v}" for k, v in self.implement.items() if not isinstance(v, dict))]
        if self.reconciliation:
            lines += ["", "## Paper vs Implement", "", *(f"* {k}: {v}" for k, v in self.reconciliation.items() if not isinstance(v, dict))]
        if self.dsr:
            lines += ["", "## Selection", "", f"* trials in the search (same universe, window and data): {self.n_trials}", f"* deflated Sharpe ratio: {self.dsr.get('dsr')} (trial variance: {self.dsr.get('var_trial_sr_source')})"]
        if self.pbo:
            lines += [f"* PBO: {self.pbo.get('pbo')}"]
        lines += ["", "## Conventions", "", *(f"* {k}: {v}" for k, v in self.conventions.items())]
        if self.links:
            lines += ["", "## Links", "", *(f"* {k}: {v}" for k, v in self.links.items())]
        if self.config:
            lines += ["", "## Config", "", "```python", self.config, "```"]
        return "\n".join(lines)


LINK_KEYS = ("portfolio_id", "port_id", "config_version", "store", "owner", "signal", "universe_sql", "price_sql", "coupon_sql", "futures_sql", "attachments", "currency", "manager", "portfolio_type")


def check_links(links: dict | None) -> dict:
    """Validate links to the operational store: known keys only (``LINK_KEYS``), strings (or, for
    ``attachments``, a list of {"name", "sha256"} entries), so a typo cannot pass as a link."""
    links = dict(links or {})
    unknown = sorted(set(links) - set(LINK_KEYS))
    if unknown:
        raise ValueError(f"unknown record link(s) {unknown}; known: {list(LINK_KEYS)}")
    for k, v in links.items():
        if k == "attachments":
            if not isinstance(v, (list, tuple)) or not all(isinstance(a, dict) and {"name", "sha256"} <= set(a) for a in v):
                raise ValueError("attachments: a list of {'name': ..., 'sha256': ...} entries (attach(path) builds one)")
            links[k] = [dict(a) for a in v]
        elif not isinstance(v, str) or not v.strip():
            raise ValueError(f"record link {k!r} must be a non-empty string, not {v!r}")
    return links


def attach(path) -> dict:
    """An attachment entry for ``links["attachments"]``: the file's name and content hash (the file
    stays in the operational store; the record pins which version it saw)."""
    p = Path(path)
    return {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _scalar(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool, type(None)))}


_ALIASES = {"sharpe": "sharpe", "sharpe_ratio": "sharpe", "ir": "information_ratio", "information_ratio": "information_ratio", "te": "tracking_error", "tracking_error": "tracking_error",
            "return": "annual_return", "annual_return": "annual_return", "active": "active_annual_return", "active_return": "active_annual_return", "active_annual_return": "active_annual_return",
            "excess": "excess_annual_return", "excess_return": "excess_annual_return", "vol": "annual_vol", "volatility": "annual_vol", "annual_vol": "annual_vol", "drawdown": "max_drawdown",
            "mdd": "max_drawdown", "max_drawdown": "max_drawdown", "turnover": "turnover", "hit_rate": "hit_rate", "dsr": "dsr", "calmar": "calmar", "sortino": "sortino", "cost_drag": "cost_drag"}
_CRITERION = re.compile(r"^\s*([A-Za-z_][A-Za-z_ ]*?)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*(%?)\s*$")


def parse_failure_criteria(text: str) -> list[tuple[str, str, float]] | None:
    """``"Sharpe < 0.3 or max_drawdown < -15%"`` -> [(statistic, op, value), ...]: the hypothesis
    fails when any criterion holds. None when the text is not machine-checkable."""
    if not text or not text.strip():
        return None
    out = []
    for part in re.split(r"\s+or\s+|;", text.strip(), flags=re.IGNORECASE):
        m = _CRITERION.match(part)
        if m is None:
            return None
        stat = _ALIASES.get(m.group(1).strip().lower().replace(" ", "_"))
        if stat is None:
            return None
        v = float(m.group(3)) / (100.0 if m.group(4) else 1.0)
        out.append((stat, m.group(2), v))
    return out


def pbo_splits(n_obs: int, min_block: int = 40) -> int:
    """CSCV blocks for PBO: an even number from 4 to 16 with at least ``min_block`` observations each
    (fewer, longer blocks on short samples, where PBO is otherwise noise)."""
    return max(4, min(16, (n_obs // min_block) // 2 * 2))


def _holds(x: float, op: str, v: float) -> bool:
    return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v}[op]


class RecordRegistry:
    """Promotion thresholds from CANDIDATE up (None: recorded, not gated): ``min_dsr``, ``max_pbo``
    (gated when two or more trials make it computable), ``min_years`` of track record; OFFICIAL
    reconciliations need ``min_reconciliation_correlation`` and ``|residual| <=
    max_reconciliation_residual``. ``anchor_path``: where the chain head is kept (put it on separate
    storage in production)."""

    def __init__(self, root: str | Path, min_dsr: float | None = 0.95, max_pbo: float | None = 0.5, min_years: float | None = 1.0, min_reconciliation_correlation: float = 0.8, max_reconciliation_residual: float = 0.02, anchor_path: str | Path | None = None, min_pbo_trials: int = 4, min_lag_stress: float | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "records.jsonl"
        self.anchor_path = Path(anchor_path) if anchor_path is not None else self.root / "records.head"
        self.min_dsr = min_dsr
        self.max_pbo = max_pbo
        self.min_years = min_years
        self.min_pbo_trials = int(min_pbo_trials)            # PBO over fewer distinct trials is too noisy to gate
        self.min_lag_stress = min_lag_stress                  # worst Sharpe at execution lags 0-3 over the strategy's own (None: not gated)
        self.min_corr = min_reconciliation_correlation
        self.max_residual = max_reconciliation_residual

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def find(self, name: str | None = None, version: int | None = None, **links) -> list[dict]:
        """Records by exact name (and version), or by a link to the operational store:
        ``find(port_id="ABC123")``."""
        out = []
        for r in self.records():
            if name is not None and r.get("name") != name:
                continue
            if version is not None and r.get("version", 1) != version:
                continue
            if any((r.get("links") or {}).get(k) != v for k, v in links.items()):
                continue
            out.append(r)
        return out

    def _anchor(self) -> dict | None:
        return json.loads(self.anchor_path.read_text()) if self.anchor_path.exists() else None

    def adopt_chain(self) -> None:
        """Anchor an existing chain (one written before anchors existed) after auditing it: the
        links must verify; from then on a removed or changed tail fails ``verify_chain``."""
        from nautilus_fi.core.fileio import atomic_write_text

        recs = self.records()
        prev = ""
        for r in recs:
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev_hash"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != r["hash"]:
                raise ValueError(f"record {r.get('record_id')} does not verify: the chain cannot be adopted")
            prev = r["hash"]
        atomic_write_text(self.anchor_path, json.dumps({"n": len(recs), "hash": prev}))

    def verify_chain(self) -> bool:
        """Every record's hash and link hold, and the chain ends where the anchor says it does."""
        prev = ""
        recs = self.records()
        for r in recs:
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev_hash"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != r["hash"]:
                return False
            prev = r["hash"]
        anchor = self._anchor()
        if anchor is None:
            return not recs
        return anchor.get("n") == len(recs) and anchor.get("hash", "") == prev

    @staticmethod
    def result_key(spec, data_hashes: dict | None, period: tuple) -> str:
        """What makes a result one result: the strategy, the data and the window."""
        return hashlib.sha256(json.dumps([spec.hash, data_hashes or {}, list(period)], sort_keys=True, default=str).encode()).hexdigest()[:20]

    def checks(self, level: RecordLevel, paper, spec, ledger, implement=None, reconciliation=None, implement_fingerprint: str | None = None, data_hashes: dict | None = None, point_in_time: bool | None = None, evidence: dict | None = None) -> dict[str, dict]:
        """Every gate for ``level``. ``implement_fingerprint`` defaults to the Implement manifest's
        spec hash; ``point_in_time`` is the caller's evidence for results that carry none (the
        legacy Paper engine), else the source's own declaration is used."""
        from nautilus_fi.kernel.result import FILL_KEYS, fill_count

        out: dict[str, dict] = {}
        order = [RecordLevel.EXPLORATORY, RecordLevel.CANDIDATE, RecordLevel.OFFICIAL]

        def put(name: str, ok: bool, detail: str, required_from: RecordLevel) -> None:
            out[name] = {"ok": bool(ok), "detail": detail, "required": order.index(level) >= order.index(required_from)}

        rets = paper.returns.iloc[1:]
        period = (str(rets.index[0].date()), str(rets.index[-1].date())) if len(rets) else ("", "")
        diag = getattr(paper, "diagnostics", None) or {}
        stats = paper.stats()

        # one record per result and level
        key = self.result_key(spec, data_hashes, period)
        dup = [r for r in self.records() if r.get("result_key") == key and r.get("level") == level.value]
        put("not_already_recorded", not dup, "first record of this result at this level" if not dup else f"already recorded as {dup[0]['record_id']}", RecordLevel.EXPLORATORY)

        # the hypothesis: registered, frozen, before the trials, with failure criteria the result does not meet
        hyp = spec.hypothesis_id
        h = ledger.hypothesis(hyp) if (ledger is not None and hyp) else None
        trials = ledger.trials(hyp) if (ledger is not None and hyp) else pd.DataFrame()
        put("hypothesis_registered", h is not None, f"{hyp or 'none'}", RecordLevel.CANDIDATE)
        put("hypothesis_frozen", h is not None and ledger.registry.frozen(hyp), "content matches its registration hash" if h is not None and ledger.registry.frozen(hyp) else "not registered through the registry, or changed since", RecordLevel.CANDIDATE)
        if h is not None and len(trials):
            first = str(trials["created_utc"].min())
            put("registered_before_trials", h.registered_utc <= first, f"registered {h.registered_utc}, first trial {first}", RecordLevel.CANDIDATE)
        else:
            put("registered_before_trials", False, "no trials recorded", RecordLevel.CANDIDATE)
        in_ledger = len(trials) > 0 and spec.hash in set(trials["spec_hash"])
        put("trial_in_ledger", in_ledger, f"{len(trials)} trials for the hypothesis", RecordLevel.CANDIDATE)

        # conventions, data and construction quality
        conv = paper.conventions
        legacy = unconfirmed_profile(conv)
        put("conventions_confirmed", conv.confirmed and legacy is None, conv.name if legacy is None else f"{conv.name}: carries the unconfirmed {legacy!r} profile's conventions", RecordLevel.CANDIDATE)
        put("official_annualisation", conv.annualisation == Annualisation.CALENDAR_365, conv.annualisation.value, RecordLevel.OFFICIAL)
        fills = fill_count(paper.missing)
        which = ", ".join(f"{k} {paper.missing[k]}" for k in FILL_KEYS if paper.missing.get(k)) + (f", cash series {paper.missing['cash_series_absent']!r} absent" if paper.missing.get("cash_series_absent") else "")
        put("no_data_fills", fills == 0, f"no filled or altered data under {paper.missing.get('policy')}" if fills == 0 else f"{fills} under {paper.missing.get('policy')}: {which.strip(', ')}", RecordLevel.CANDIDATE)
        hard = int((paper.violations["severity"] == "HARD").sum()) if not paper.violations.empty and "severity" in paper.violations else 0
        put("no_hard_constraint_violations", hard == 0, f"{hard} hard violations", RecordLevel.CANDIDATE)
        relaxed = int(diag.get("relaxed_rebalances", 0)) + int(diag.get("infeasible_rebalances", 0))
        put("no_relaxed_constructions", relaxed == 0, f"{relaxed} rebalances relaxed or left unchanged by an infeasible construction", RecordLevel.CANDIDATE)
        la_on = bool(diag.get("lookahead_checked", True))
        la_bad = [v for v in diag.get("lookahead", []) if isinstance(v, dict) and v.get("violation")]
        put("lookahead_checked", la_on and not la_bad, "structural and empirical checks passed" if la_on and not la_bad else ("checks switched off" if not la_on else la_bad[0]["violation"]), RecordLevel.CANDIDATE)
        years = float(stats.get("years") or 0.0)
        if self.min_years is not None:
            put("track_record", years >= self.min_years, f"{years:.2f} years, minimum {self.min_years:g}", RecordLevel.CANDIDATE)

        # selection: the whole search on this universe, window and data
        dsr = pbo = None
        if ledger is not None:
            scope = ledger.scope(getattr(spec, "universe", "") or "", period, getattr(spec, "data_version", "") or "")
            if not any(t["spec_hash"] == spec.hash for t in scope):
                scope = scope + [t for t in ledger.counted() if t["spec_hash"] == spec.hash][-1:]
            ppy = paper.conventions.periods_per_year(rets.index)
            dsr = ledger.deflated_sharpe(None, rets, ppy, trials=scope)
            hyps = sorted({str(t.get("hypothesis_id")) for t in scope})
            gated = "" if self.min_dsr is not None else ", not thresholded"
            put("dsr_from_ledger", dsr.get("var_trial_sr_source") in ("supplied", "single trial"), f"DSR {dsr.get('dsr'):.3f} over {dsr.get('ledger_trials')} trials in the search ({len(hyps)} hypothesis labels){gated}", RecordLevel.CANDIDATE)
            if self.min_dsr is not None:
                value = float(dsr.get("dsr", float("nan")))
                put("dsr_threshold", value >= self.min_dsr, f"DSR {value:.3f}, minimum {self.min_dsr:g}", RecordLevel.CANDIDATE)
            pbo = ledger.pbo(None, n_splits=pbo_splits(len(rets)), trials=scope)
            if self.max_pbo is not None:
                value = float(pbo.get("pbo", float("nan")))
                enough = int(pbo.get("n_trials") or 0) >= self.min_pbo_trials
                known = not math.isnan(value)
                ok = (value <= self.max_pbo) if (known and enough) else True
                detail = f"PBO {value:.3f} over {pbo.get('n_trials')} trials, maximum {self.max_pbo:g}" if known else f"PBO not computable ({pbo.get('note', '')})"
                put("pbo_threshold", ok, detail + ("" if enough or not known else f" (not gated below {self.min_pbo_trials} trials)"), RecordLevel.CANDIDATE)
            if hyp:
                used = trials[trials["uses_holdout"].astype(bool)] if len(trials) and "uses_holdout" in trials else pd.DataFrame()
                if level == RecordLevel.OFFICIAL:
                    others = sorted(set(used["spec_hash"]) - {spec.hash}) if len(used) else []
                    put("holdout_guard", not others, "holdout used only by this spec" if not others else f"holdout also used by {len(others)} other spec(s): {others[:3]}", RecordLevel.CANDIDATE)
                else:
                    put("holdout_guard", used.empty, "holdout untouched" if used.empty else f"holdout viewed by {len(used)} trial(s)", RecordLevel.CANDIDATE)
        else:
            put("dsr_from_ledger", False, "no ledger", RecordLevel.CANDIDATE)
        if h is not None:
            crit = parse_failure_criteria(h.failure_criteria)
            if crit is None:
                put("failure_criteria", False, f"not machine-checkable: {h.failure_criteria!r} (write '<statistic> <op> <number>' joined by 'or', e.g. 'sharpe < 0.3 or max_drawdown < -15%')", RecordLevel.CANDIDATE)
            else:
                values = {**stats, "dsr": (dsr or {}).get("dsr")}
                unknown = [c for c, _, _ in crit if values.get(c) is None or not math.isfinite(float(values[c]))]
                met = [f"{c} {op} {v:g} ({float(values[c]):.4g})" for c, op, v in crit if c not in unknown and _holds(float(values[c]), op, v)]
                ok = not met and not unknown
                put("failure_criteria", ok, "none met: " + h.failure_criteria if ok else (f"met: {'; '.join(met)}" if met else f"cannot evaluate {unknown} on this result"), RecordLevel.CANDIDATE)

        if self.min_lag_stress is not None:
            rows = (evidence or {}).get("lag_stress") or []
            ratios = [float(r["ratio_to_own_lag"]) for r in rows if r.get("ratio_to_own_lag") is not None and math.isfinite(float(r["ratio_to_own_lag"]))]
            worst = min(ratios) if ratios else float("nan")
            put("lag_stress", bool(ratios) and worst >= self.min_lag_stress, f"worst Sharpe at lags {[r['lag'] for r in rows]} is {worst:.2f}x its own (minimum {self.min_lag_stress:g})" if ratios else "no lag-stress evidence (Session.lag_stress)", RecordLevel.CANDIDATE)

        # OFFICIAL: point-in-time inputs and an Implement run of the same strategy, reconciled
        pit = (diag.get("data") or {}).get("point_in_time", point_in_time)
        if implement is not None and implement_fingerprint is None and getattr(implement, "manifest", None) is not None:
            implement_fingerprint = (implement.manifest.spec or {}).get("hash")
        put("point_in_time_inputs", pit is True, "the source declares knowledge-time (point-in-time) reads" if pit is True else f"not shown point in time ({pit})", RecordLevel.OFFICIAL)
        if implement is not None:
            st, oc = implement.status, implement.outcome
            hashes = bool(implement.manifest and implement.manifest.output_hashes.get("event_chain"))
            put("implement_completed", st == "COMPLETED" and oc == "SOLVENT", f"{st} / {oc}", RecordLevel.OFFICIAL)
            put("implement_manifest", hashes, "output and event-chain hashes" if hashes else "missing", RecordLevel.OFFICIAL)
            put("implement_matches_paper", implement_fingerprint == spec.hash, "same strategy" if implement_fingerprint == spec.hash else f"Implement ran {implement_fingerprint or 'an unidentified strategy'}, not {spec.hash}", RecordLevel.OFFICIAL)
        else:
            put("implement_completed", False, "no Implement run", RecordLevel.OFFICIAL)
            put("implement_manifest", False, "no Implement run", RecordLevel.OFFICIAL)
            put("implement_matches_paper", False, "no Implement run", RecordLevel.OFFICIAL)
        need = ("paper_gross_total", "implemented_total", "holdings_gap", "execution", "residual", "daily_correlation")
        rec = reconciliation or {}
        missing_keys = [k for k in need if not isinstance(rec.get(k), (int, float)) or not math.isfinite(float(rec[k]))]
        put("reconciled", not missing_keys, "Paper vs Implement decomposition" if not missing_keys else f"missing {missing_keys}", RecordLevel.OFFICIAL)
        if not missing_keys:
            corr, resid = float(rec["daily_correlation"]), float(rec["residual"])
            put("reconciliation_tolerance", corr >= self.min_corr and abs(resid) <= self.max_residual, f"daily correlation {corr:.3f} (min {self.min_corr:g}), residual {resid:.4%} (max {self.max_residual:.2%})", RecordLevel.OFFICIAL)
        else:
            put("reconciliation_tolerance", False, "no reconciliation", RecordLevel.OFFICIAL)
        return out

    def submit(self, level: RecordLevel, paper, spec, ledger=None, implement=None, reconciliation: dict | None = None, data_hashes: dict | None = None, name: str | None = None, implement_fingerprint: str | None = None, point_in_time: bool | None = None, evidence: dict | None = None, links: dict | None = None, config: str | None = None) -> OfficialRecord:
        """``links``: ids in the operational store (``LINK_KEYS``); ``config``: the strategy as
        code. Records are create-only and versioned by exact name: the n-th record named X has
        ``version`` n, and nothing is ever rewritten."""
        from nautilus_fi.core.fileio import atomic_write_text, file_lock

        links = check_links(links)

        with file_lock(self.root / "records.lock"):                  # gates, chain head read and append are one step
            if self.records() and not self.verify_chain():
                raise GateFailure(level, {"chain": "the record chain does not verify against its anchor (a record was changed or removed, or the chain predates anchors): audit it, then adopt_chain()"})
            checks = self.checks(level, paper, spec, ledger, implement, reconciliation, implement_fingerprint, data_hashes, point_in_time, evidence)
            failed = {k: v["detail"] for k, v in checks.items() if v["required"] and not v["ok"]}
            if failed:
                raise GateFailure(level, failed)
            hyp = spec.hypothesis_id
            rets = paper.returns.iloc[1:]
            period = (str(rets.index[0].date()), str(rets.index[-1].date()))
            dsr = pbo = None
            n = 0
            if ledger is not None:
                scope = ledger.scope(getattr(spec, "universe", "") or "", period, getattr(spec, "data_version", "") or "")
                dsr = ledger.deflated_sharpe(None, rets, paper.conventions.periods_per_year(rets.index), trials=scope)
                pbo = ledger.pbo(None, n_splits=pbo_splits(len(rets)), trials=scope)
                n = len(scope)
            prev = self.records()[-1]["hash"] if self.records() else ""
            rec_name = name or paper.name
            version = 1 + sum(1 for r in self.records() if r.get("name") == rec_name)
            rid = hashlib.sha256(f"{spec.hash}{level.value}{datetime.now(UTC).isoformat()}".encode()).hexdigest()[:12]
            rec = OfficialRecord(rid, level.value, rec_name, hyp, spec.hash, paper.conventions.to_dict(), _scalar(paper.stats()), _scalar(implement.stats()) if implement is not None else None, reconciliation, None if dsr is None else _scalar(dsr), None if pbo is None else _scalar(pbo), n, checks, data_hashes or {}, implement.manifest.code_version if implement is not None and implement.manifest else None, prev_hash=prev, result_key=self.result_key(spec, data_hashes, period), thresholds={"min_dsr": self.min_dsr, "max_pbo": self.max_pbo, "min_years": self.min_years, "min_reconciliation_correlation": self.min_corr, "max_reconciliation_residual": self.max_residual, "min_lag_stress": self.min_lag_stress}, links=links, config=config, version=version)
            body = {k: v for k, v in asdict(rec).items() if k != "hash"}
            rec.hash = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(rec), default=str) + "\n")
            atomic_write_text(self.anchor_path, json.dumps({"n": len(self.records()), "hash": rec.hash}))
            atomic_write_text(self.root / f"{rid}.md", rec.to_markdown())
        return rec


__all__ = ["LINK_KEYS", "GateFailure", "OfficialRecord", "RecordLevel", "RecordRegistry", "attach", "check_links", "parse_failure_criteria"]
