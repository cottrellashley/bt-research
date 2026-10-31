"""Portfolio optimiser: LP, MILP and QP over trades, solved by HiGHS (``highspy``, MIT licence).

Decision variables are buys ``b`` and sells ``s`` per name (``w = w0 + b - s``), so weight bounds
become variable bounds, turnover and transaction costs are linear, and a 9,000-name rebalance stays
a sparse problem. Objective terms:

* ``alpha``      maximise ``scores . w``
* ``cost``       minimise ``sum cost_i (b_i + s_i)`` (one-way cost per unit traded)
* ``tracking``   minimise the L1 norm of active bucket exposures (e.g. OASD and DTS by sector x
                 rating x maturity), each field scaled by the benchmark's total exposure
* ``active_l1``  minimise ``sum |w_i - wb_i|``
* ``risk``       minimise ``aversion x`` (active) variance from a factor model (QP)
* ``cvar``       minimise ``weight x`` CVaR of the (active) loss over scenarios (Rockafellar-Uryasev LP)

Constraints come from ``construct.grammar``. Hard constraints bind; soft ones carry slack at their
penalty and are reported as relaxations. Every solve reports status, objective components,
binding constraints, duals (continuous problems), turnover and solve time; an infeasible or failed
solve returns the current weights with ``ok=False`` instead of a fabricated portfolio.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import highspy
import numpy as np
import pandas as pd
from scipy import sparse

from nautilus_fi.construct.cross_section import CrossSection
from nautilus_fi.construct.grammar import ConstraintSpec, parse_constraints
from nautilus_fi.domain.conventions import turnover_all

INF = highspy.kHighsInf


@dataclass
class TrackingTerm:
    fields: tuple[str, ...] = ("oasd", "dts")
    by: tuple[str, ...] = ("sector",)
    weight: float = 1.0
    scale: dict | None = None                    # field -> scale (default: benchmark total |exposure|)


@dataclass
class RiskModel:
    loadings: pd.DataFrame                       # ids x factors
    factor_cov: pd.DataFrame                     # factors x factors (per-period variance units)
    specific_var: pd.Series                      # ids
    aversion: float = 1.0
    active: bool = True


@dataclass
class CVaRTerm:
    beta: float = 0.95
    weight: float = 1.0
    active: bool = True


@dataclass
class Objective:
    alpha: float = 1.0
    cost: float = 1.0
    tracking: TrackingTerm | None = None
    active_l1: float = 0.0
    risk: RiskModel | None = None
    cvar: CVaRTerm | None = None

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "cost": self.cost, "tracking": None if self.tracking is None else {"fields": list(self.tracking.fields), "by": list(self.tracking.by), "weight": self.tracking.weight}, "active_l1": self.active_l1, "risk": None if self.risk is None else {"aversion": self.risk.aversion, "active": self.risk.active, "factors": list(self.risk.factor_cov.columns)}, "cvar": None if self.cvar is None else {"beta": self.cvar.beta, "weight": self.cvar.weight, "active": self.cvar.active}}


@dataclass
class OptimiserSettings:
    time_limit: float = 60.0
    mip_rel_gap: float = 1e-4
    threads: int | None = None
    verbose: bool = False
    zero_tol: float = 1e-9
    on_infeasible: str = "relax"                 # relax: retry with conflicting limits soft (recorded); keep: hold current weights
    relax_penalty: float = 1_000.0
    initial_build_turnover_exempt: bool = True   # turnover limits do not apply when building from cash
    mip_strategy: str = "auto"                   # full | lp_round | auto (lp_round above mip_full_max_names)
    mip_full_max_names: int = 1500
    cardinality_shrink: float = 0.7              # lp_round: share of candidates kept per elimination round


@dataclass
class OptimisationResult:
    weights: pd.Series
    ok: bool
    status: str
    objective: float
    components: dict
    binding: list[str]
    duals: dict
    relaxations: list[dict]
    turnover: dict
    n_names: int
    solve_seconds: float
    problem: dict = field(default_factory=dict)
    message: str = ""

    def summary(self) -> dict:
        return {"ok": self.ok, "status": self.status, "objective": self.objective, **{f"obj_{k}": v for k, v in self.components.items()}, **{f"turnover_{k}": v for k, v in self.turnover.items()}, "n_names": self.n_names, "n_binding": len(self.binding), "n_relaxed": len(self.relaxations), "solve_seconds": self.solve_seconds, **{f"problem_{k}": v for k, v in self.problem.items()}}


class OptimisationError(RuntimeError):
    pass


class _Model:
    def __init__(self) -> None:
        self.lo: list[np.ndarray] = []
        self.hi: list[np.ndarray] = []
        self.cost: list[np.ndarray] = []
        self.integer: list[np.ndarray] = []
        self.n = 0
        self.r: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        self.row_lo: list[float] = []
        self.row_hi: list[float] = []
        self.row_name: list[str] = []
        self.row_soft: dict[int, tuple[int, int, str]] = {}
        self.q: dict[tuple[int, int], float] = defaultdict(float)
        self.offset = 0.0

    def add_vars(self, n: int, lo, hi, cost=0.0, integer: bool = False) -> np.ndarray:
        idx = np.arange(self.n, self.n + n)
        self.lo.append(np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy())
        self.hi.append(np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy())
        self.cost.append(np.broadcast_to(np.asarray(cost, dtype=float), (n,)).copy())
        self.integer.append(np.full(n, integer))
        self.n += n
        return idx

    def add_row(self, cols, vals, lo: float, hi: float, name: str) -> int:
        k = len(self.row_lo)
        cols = np.asarray(cols, dtype=np.int64)
        self.r.append(np.full(len(cols), k, dtype=np.int64))
        self.c.append(cols)
        self.v.append(np.asarray(vals, dtype=float))
        self.row_lo.append(-INF if lo is None or lo == -np.inf else float(lo))
        self.row_hi.append(INF if hi is None or hi == np.inf else float(hi))
        self.row_name.append(name)
        return k

    def add_rows(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, lo: np.ndarray, hi: np.ndarray, names: list[str]) -> np.ndarray:
        """Vectorised rows: ``rows`` are local row indices 0..m-1."""
        k0 = len(self.row_lo)
        self.r.append(np.asarray(rows, dtype=np.int64) + k0)
        self.c.append(np.asarray(cols, dtype=np.int64))
        self.v.append(np.asarray(vals, dtype=float))
        self.row_lo += [(-INF if not np.isfinite(x) else float(x)) for x in lo]
        self.row_hi += [(INF if not np.isfinite(x) else float(x)) for x in hi]
        self.row_name += list(names)
        return np.arange(k0, k0 + len(lo))

    def soften(self, row: int, penalty: float, label: str) -> None:
        up = self.add_vars(1, 0.0, INF, penalty)[0]
        dn = self.add_vars(1, 0.0, INF, penalty)[0]
        self.r.append(np.array([row, row], dtype=np.int64))
        self.c.append(np.array([up, dn], dtype=np.int64))
        self.v.append(np.array([1.0, -1.0]))
        self.row_soft[row] = (up, dn, label)

    def add_q(self, i: int, j: int, v: float) -> None:
        a, b = (i, j) if i >= j else (j, i)
        self.q[(a, b)] += v

    def solve(self, settings: OptimiserSettings) -> tuple[str, np.ndarray, np.ndarray, float, float, highspy.Highs]:
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(settings.verbose))
        h.setOptionValue("time_limit", float(settings.time_limit))
        h.setOptionValue("mip_rel_gap", float(settings.mip_rel_gap))
        if settings.threads:
            h.setOptionValue("threads", int(settings.threads))
        lo, hi, cost, integer = (np.concatenate(x) if x else np.zeros(0) for x in (self.lo, self.hi, self.cost, self.integer))
        m = len(self.row_lo)
        A = sparse.coo_matrix((np.concatenate(self.v) if self.v else np.zeros(0), (np.concatenate(self.r) if self.r else np.zeros(0, dtype=np.int64), np.concatenate(self.c) if self.c else np.zeros(0, dtype=np.int64))), shape=(m, self.n)).tocsc()
        A.sum_duplicates()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.n, m
        lp.col_cost_ = cost
        lp.col_lower_ = np.where(np.isfinite(lo), lo, -INF)
        lp.col_upper_ = np.where(np.isfinite(hi), hi, INF)
        lp.row_lower_ = np.array(self.row_lo, dtype=float)
        lp.row_upper_ = np.array(self.row_hi, dtype=float)
        lp.offset_ = self.offset
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data
        if integer.any():
            lp.integrality_ = [highspy.HighsVarType.kInteger if x else highspy.HighsVarType.kContinuous for x in integer]
        h.passModel(lp)
        if self.q:
            keys = sorted(self.q, key=lambda ij: (ij[1], ij[0]))          # column-wise lower triangle
            cols = np.array([j for _, j in keys])
            start = np.searchsorted(cols, np.arange(self.n + 1))
            hess = highspy.HighsHessian()
            hess.dim_ = self.n
            hess.format_ = highspy.HessianFormat.kTriangular
            hess.start_ = start.astype(np.int32)
            hess.index_ = np.array([i for i, _ in keys], dtype=np.int32)
            hess.value_ = np.array([self.q[k] for k in keys], dtype=float)
            h.passHessian(hess)
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        sol = h.getSolution()
        x = np.array(sol.col_value, dtype=float) if len(sol.col_value) else np.zeros(self.n)
        duals = np.array(sol.row_dual, dtype=float) if len(sol.row_dual) else np.zeros(m)
        info = h.getInfo()
        return status, x, duals, float(info.objective_function_value), float(getattr(info, "mip_gap", 0.0) or 0.0), A


def optimise(cs: CrossSection, objective: Objective | None = None, constraints=None, long_only: bool = True, settings: OptimiserSettings | None = None) -> OptimisationResult:
    """Solve the construction problem. Mixed-integer constraints (cardinality, minimum holdings) on
    large universes use ``lp_round``: solve the LP relaxation, shrink the candidate set by successive
    LPs until the cardinality bound can bind, fix binaries the LP already decides, and leave binaries
    only for names the LP holds below their minimum."""
    settings = settings or OptimiserSettings()
    cons = parse_constraints(constraints)
    notes: list[dict] = []
    if settings.initial_build_turnover_exempt and float(cs.current.abs().sum()) < 1e-9 and any(c.kind == "turnover" for c in cons):
        cons = [c for c in cons if c.kind != "turnover"]
        notes.append({"constraint": "turnover", "group": None, "reason": "initial build from cash: turnover limit not applied"})
    integer_kinds = [c for c in cons if c.kind in ("cardinality", "min_holding")]
    strategy = settings.mip_strategy if settings.mip_strategy != "auto" else ("lp_round" if len(cs.ids) > settings.mip_full_max_names else "full")
    run = (lambda cc: _optimise_lp_round(cs, objective, cc, long_only, settings)) if integer_kinds and strategy == "lp_round" else (lambda cc: _optimise(cs, objective, cc, long_only, settings))
    res = run(cons)
    if not res.ok and settings.on_infeasible == "relax" and res.status in ("Infeasible", "Primal infeasible or unbounded"):
        from dataclasses import replace as _replace

        keep_hard = ("budget", "cash", "weight", "cardinality", "min_holding")
        soft = [c if (c.kind in keep_hard or not c.hard) else _replace(c, hard=False, penalty=settings.relax_penalty) for c in cons]
        retry = run(soft)
        if retry.ok:
            retry.relaxations = [{"constraint": "all limits", "group": None, "reason": f"hard problem {res.status}: limits other than budget / weight / integer made soft at penalty {settings.relax_penalty}"}, *retry.relaxations]
            retry.message = f"relaxed after {res.status}"
            res = retry
    res.relaxations = notes + res.relaxations
    return res


def _optimise_lp_round(cs: CrossSection, objective, cons: list[ConstraintSpec], long_only: bool, settings: OptimiserSettings) -> OptimisationResult:
    """Large-universe heuristic for cardinality / minimum holdings: LP relaxation, successive LPs that
    force the smallest names out until the cardinality bound holds, then round each name held below
    its minimum up (at least half the minimum) or out, and re-solve the LP with those decisions fixed.
    Falls back to the full MILP (time-limited) on the remaining fractional names if rounding is
    infeasible."""
    from dataclasses import replace

    t0 = time.perf_counter()
    full = replace(settings, mip_strategy="full")
    relaxed = [c for c in cons if c.kind not in ("cardinality", "min_holding")]
    card = next((c for c in cons if c.kind == "cardinality" and c.max is not None), None)
    n = len(cs.ids)
    zero_mask = np.zeros(n, dtype=bool)
    lp = _optimise(cs, objective, relaxed, long_only, full)
    if not lp.ok:
        return lp
    rounds = 0
    w = lp.weights.reindex(cs.ids).fillna(0.0).to_numpy()
    while card is not None and int((np.abs(w) > settings.zero_tol).sum()) > card.max and rounds < 20:
        held = np.nonzero(np.abs(w) > settings.zero_tol)[0]
        keep_n = max(int(card.max), int(np.ceil(len(held) * settings.cardinality_shrink)))
        order = held[np.argsort(-np.abs(w[held]), kind="stable")]
        zero_mask[:] = True
        zero_mask[order[:keep_n]] = False
        lp = _optimise(cs, objective, relaxed, long_only, full, ub_override=np.where(zero_mask, 0.0, np.inf))
        if not lp.ok:
            return lp
        w = lp.weights.reindex(cs.ids).fillna(0.0).to_numpy()
        rounds += 1
    mins = np.zeros(n)
    for c in cons:
        if c.kind == "min_holding":
            mins = np.full(n, c.min) if c.min is not None else (cs.min_weight.to_numpy(dtype=float) if cs.min_weight is not None else np.zeros(n))
    held = np.abs(w) > settings.zero_tol
    frac = held & (w < mins - 1e-12)
    up = frac & (w >= 0.5 * mins)
    out = (frac & ~up) | ~held | zero_mask
    lb_o = np.where(held & ~out, mins, 0.0)
    ub_o = np.where(out, 0.0, np.inf)
    res = _optimise(cs, objective, relaxed + ([ConstraintSpec("cardinality", card.min, card.max)] if card is not None and card.min is not None else []), long_only, full, lb_override=lb_o, ub_override=ub_o) if not (card is not None and card.min is not None) else None
    strategy = "lp_round"
    if res is None or not res.ok:
        strategy = "lp_round+milp"
        z_lo = np.where(held & ~frac & ~zero_mask, 1.0, 0.0)
        z_hi = np.where(zero_mask | ~held, 0.0, 1.0)
        res = _optimise(cs, objective, cons, long_only, full, z_fix=(z_lo, z_hi))
    res.problem.update({"mip_strategy": strategy, "elimination_rounds": rounds, "rounded_up": int(up.sum()), "rounded_out": int((frac & ~up).sum()), "fixed_out": int(out.sum())})
    res.solve_seconds = time.perf_counter() - t0
    return res


def _optimise(cs: CrossSection, objective: Objective | None, cons: list[ConstraintSpec], long_only: bool, settings: OptimiserSettings, z_fix: tuple[np.ndarray, np.ndarray] | None = None, lb_override: np.ndarray | None = None, ub_override: np.ndarray | None = None) -> OptimisationResult:
    t0 = time.perf_counter()
    objective = objective or Objective()
    ids = cs.ids
    n = len(ids)
    w0 = cs.current.to_numpy(dtype=float)
    wb = cs.benchmark.to_numpy(dtype=float)
    alpha = (cs.scores.fillna(0.0).to_numpy(dtype=float) if cs.scores is not None else np.zeros(n)) * objective.alpha
    unit_cost = (cs.cost_bp.fillna(cs.cost_bp.median() if cs.cost_bp.notna().any() else 0.0).to_numpy(dtype=float) / 1e4 if cs.cost_bp is not None else np.zeros(n)) * objective.cost
    elig = cs.eligible.to_numpy(dtype=bool)
    data = cs.data

    # ---------------------------------------------------------------- weight bounds -> trade bounds
    lb = np.zeros(n) if long_only else np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    for c in cons:
        if c.kind == "weight" and c.hard:
            if c.min is not None:
                lb = np.maximum(lb, c.min)
            if c.max is not None:
                ub = np.minimum(ub, c.max)
        elif c.kind == "active_weight" and c.hard:
            if c.min is not None:
                lb = np.maximum(lb, wb + c.min)
            if c.max is not None:
                ub = np.minimum(ub, wb + c.max)
    ub = np.where(elig, ub, np.minimum(ub, np.maximum(w0, 0.0)))               # ineligible names can only shrink
    if ub_override is not None:
        ub = np.minimum(ub, ub_override)
    if lb_override is not None:
        lb = np.maximum(lb, lb_override)
    lb = np.where(elig, lb, np.minimum(lb, 0.0) if long_only else np.minimum(lb, np.minimum(w0, 0.0)))
    clash = lb > ub
    relaxations: list[dict] = [{"constraint": "weight bounds", "group": ids[i], "from": float(lb[i]), "to": float(ub[i]), "reason": "lower bound above upper bound (ineligible or conflicting)"} for i in np.nonzero(clash)[0][:50]]
    lb = np.where(clash, ub, lb)
    b_hi = np.maximum(ub - w0, 0.0)
    s_hi = np.maximum(w0 - lb, 0.0)
    b_lo = np.maximum(lb - w0, 0.0)
    s_lo = np.maximum(w0 - ub, 0.0)
    b_hi = np.where(np.isfinite(b_hi), b_hi, INF)
    s_hi = np.where(np.isfinite(s_hi), s_hi, INF)

    M = _Model()
    eps = 1e-9                                                          # breaks buy/sell degeneracy when costs are zero
    b = M.add_vars(n, b_lo, b_hi, -alpha + unit_cost + eps)
    s = M.add_vars(n, s_lo, s_hi, alpha + unit_cost + eps)
    M.offset -= float(alpha @ w0)
    both = np.concatenate([b, s])

    def trade_coefs(x: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(n) if mask is None else np.nonzero(mask)[0]
        return np.concatenate([b[idx], s[idx]]), np.concatenate([x[idx], -x[idx]])

    def add_band(name: str, x: np.ndarray, const: float, lo: float | None, hi: float | None, c: ConstraintSpec, mask: np.ndarray | None = None) -> None:
        cols, vals = trade_coefs(x, mask)
        row = M.add_row(cols, vals, -INF if lo is None else lo - const, INF if hi is None else hi - const, name)
        if not c.hard:
            M.soften(row, c.penalty, name)

    has_budget = any(c.kind in ("budget", "cash") for c in cons)
    if not has_budget and long_only:
        cons = [*cons, ConstraintSpec("budget", 1.0, 1.0, name="budget (default: fully invested)")]
    ones = np.ones(n)
    integer_needed = False
    for c in cons:
        k = c.kind
        if k == "budget":
            add_band(c.label, ones, float(w0.sum()), c.min, c.max, c)
        elif k == "cash":
            add_band(c.label, ones, float(w0.sum()), None if c.max is None else 1.0 - c.max, None if c.min is None else 1.0 - c.min, c)
        elif k in ("weight", "active_weight") and not c.hard:
            for i in range(n):
                base = wb[i] if k == "active_weight" else 0.0
                lo_i = None if c.min is None else base + c.min
                hi_i = None if c.max is None else base + c.max
                mask = np.zeros(n, dtype=bool)
                mask[i] = True
                add_band(f"{c.label}:{ids[i]}", ones, float(w0[i]), lo_i, hi_i, c, mask)
        elif k in ("group", "active_group"):
            g = data[c.attribute].astype(str).to_numpy()
            for grp in (np.unique(g) if c.value is None else [str(c.value)]):
                mask = g == grp
                if not mask.any():
                    continue
                const = float(w0[mask].sum() - (wb[mask].sum() if k == "active_group" else 0.0))
                add_band(f"{c.label}:{grp}", ones, const, c.min, c.max, c, mask)
        elif k in ("exposure", "active_exposure"):
            x = data[c.attribute].astype(float).fillna(0.0).to_numpy()
            keys = data[c.group_by].astype(str).to_numpy() if c.group_by else np.full(n, "ALL")
            for grp in np.unique(keys):
                mask = keys == grp
                bexp = float((wb[mask] * x[mask]).sum())
                const = float((w0[mask] * x[mask]).sum() - (bexp if k == "active_exposure" else 0.0))
                lo, hi = c.min, c.max
                if k == "active_exposure" and c.relative:
                    lo = None if lo is None else lo * abs(bexp)
                    hi = None if hi is None else hi * abs(bexp)
                add_band(f"{c.label}{'' if grp == 'ALL' else ':' + grp}", x, const, lo, hi, c, mask)
        elif k == "turnover":
            fb, fs = {"HALF_SUM": (0.5, 0.5), "TWO_WAY": (1.0, 1.0), "ONE_WAY_BUYS": (1.0, 0.0)}[c.definition]
            row = M.add_row(both, np.concatenate([np.full(n, fb), np.full(n, fs)]), -INF, c.max, c.label)
            if not c.hard:
                M.soften(row, c.penalty, c.label)
        elif k == "leverage":
            if long_only:
                add_band(c.label, ones, float(w0.sum()), None, c.max, c)
            else:
                p = M.add_vars(n, 0.0, INF)
                # p_i >= |w0_i + b_i - s_i|
                r = np.arange(n)
                M.add_rows(np.concatenate([r, r, r]), np.concatenate([p, b, s]), np.concatenate([np.ones(n), -np.ones(n), np.ones(n)]), w0, np.full(n, np.inf), [f"abs+:{i}" for i in ids])
                M.add_rows(np.concatenate([r, r, r]), np.concatenate([p, b, s]), np.concatenate([np.ones(n), np.ones(n), -np.ones(n)]), -w0, np.full(n, np.inf), [f"abs-:{i}" for i in ids])
                row = M.add_row(p, np.ones(n), -INF, c.max, c.label)
                if not c.hard:
                    M.soften(row, c.penalty, c.label)
        elif k in ("cardinality", "min_holding"):
            integer_needed = True
        elif k == "cvar":
            _add_cvar(M, cs, b, s, w0, wb, n, c.beta, c.active, constraint=c)

    # ---------------------------------------------------------------- mixed-integer: cardinality and minimum holdings
    z = None
    if integer_needed:
        if not long_only:
            raise OptimisationError("cardinality / min_holding constraints are implemented for long-only portfolios")
        cap = np.where(np.isfinite(ub), ub, 1.0)
        z = M.add_vars(n, 0.0 if z_fix is None else z_fix[0], 1.0 if z_fix is None else z_fix[1], 0.0, integer=True)
        r = np.arange(n)
        # w0 + b - s - cap z <= 0
        M.add_rows(np.concatenate([r, r, r]), np.concatenate([b, s, z]), np.concatenate([np.ones(n), -np.ones(n), -cap]), np.full(n, -np.inf), -w0, [f"link:{i}" for i in ids])
        for c in cons:
            if c.kind == "cardinality":
                row = M.add_row(z, np.ones(n), -INF if c.min is None else c.min, INF if c.max is None else c.max, c.label)
                if not c.hard:
                    M.soften(row, c.penalty, c.label)
            elif c.kind == "min_holding":
                mw = np.full(n, c.min) if c.min is not None else (cs.min_weight.to_numpy(dtype=float) if cs.min_weight is not None else np.zeros(n))
                mw = np.minimum(mw, cap)
                # w0 + b - s - m z >= 0
                M.add_rows(np.concatenate([r, r, r]), np.concatenate([b, s, z]), np.concatenate([np.ones(n), -np.ones(n), -mw]), -w0, np.full(n, np.inf), [f"{c.label}:{i}" for i in ids])

    # ---------------------------------------------------------------- objective extras
    track_cols: list[tuple[int, str, float]] = []
    if objective.tracking is not None:
        tt = objective.tracking
        by = [b_ for b_ in tt.by if b_ in data]
        key = data[by].astype(str).agg("|".join, axis=1).to_numpy() if by else np.full(n, "ALL")
        for f in tt.fields:
            if f not in data:
                continue
            x = data[f].astype(float).fillna(0.0).to_numpy()
            scale = (tt.scale or {}).get(f) or max(float(np.abs(wb * x).sum()), 1e-9)
            for grp in np.unique(key):
                mask = key == grp
                if not (np.abs(x[mask]).sum() > 0 and (mask & ((wb > 0) | (w0 != 0) | elig)).any()):
                    continue
                const = float(((w0 - wb)[mask] * x[mask]).sum())
                u = M.add_vars(1, 0.0, INF, tt.weight / scale)[0]
                cols, vals = trade_coefs(x, mask)
                M.add_row(np.concatenate([[u], cols]), np.concatenate([[1.0], -vals]), const, INF, f"track+:{f}:{grp}")
                M.add_row(np.concatenate([[u], cols]), np.concatenate([[1.0], vals]), -const, INF, f"track-:{f}:{grp}")
                track_cols.append((u, f, tt.weight / scale))
    l1_cols = None
    if objective.active_l1 > 0:
        a = M.add_vars(n, 0.0, INF, objective.active_l1)
        r = np.arange(n)
        d = w0 - wb
        M.add_rows(np.concatenate([r, r, r]), np.concatenate([a, b, s]), np.concatenate([np.ones(n), -np.ones(n), np.ones(n)]), d, np.full(n, np.inf), [f"l1+:{i}" for i in ids])
        M.add_rows(np.concatenate([r, r, r]), np.concatenate([a, b, s]), np.concatenate([np.ones(n), np.ones(n), -np.ones(n)]), -d, np.full(n, np.inf), [f"l1-:{i}" for i in ids])
        l1_cols = a
    cvar_cols = None
    if objective.cvar is not None:
        cvar_cols = _add_cvar(M, cs, b, s, w0, wb, n, objective.cvar.beta, objective.cvar.active, weight=objective.cvar.weight)
    risk_info = None
    if objective.risk is not None:
        risk_info = _add_risk(M, objective.risk, ids, b, s, w0, wb, n)

    status, x, duals, obj, gap, A = M.solve(settings)
    ok = status in ("Optimal",) or (status in ("Time limit reached", "Iteration limit reached") and np.isfinite(obj) and len(x) == M.n and obj < INF)
    solve_s = time.perf_counter() - t0
    problem = {"n_names": n, "n_vars": M.n, "n_rows": len(M.row_lo), "integer": bool(integer_needed), "quadratic": bool(M.q), "mip_gap": gap}
    if not ok or len(x) != M.n:
        return OptimisationResult(cs.current.copy(), False, status, float("nan"), {}, [], {}, relaxations, {k: 0.0 for k in ("ONE_WAY_BUYS", "HALF_SUM", "TWO_WAY")}, int((np.abs(w0) > settings.zero_tol).sum()), solve_s, problem, f"solver status {status}: weights left unchanged")
    w = w0 + x[b] - x[s]
    w[np.abs(w) < settings.zero_tol] = 0.0
    weights = pd.Series(w, index=ids)
    act = A @ x
    binding = [M.row_name[i] for i in range(len(M.row_lo)) if not M.row_name[i].startswith(("track", "l1", "abs", "link", "cvar")) and ((M.row_lo[i] > -INF and abs(act[i] - M.row_lo[i]) < 1e-7) or (M.row_hi[i] < INF and abs(act[i] - M.row_hi[i]) < 1e-7))]
    dual_map = {} if integer_needed else {M.row_name[i]: float(duals[i]) for i in range(len(M.row_lo)) if abs(duals[i]) > 1e-10 and not M.row_name[i].startswith(("track", "l1", "abs", "cvar"))}
    for row, (up, dn, label) in M.row_soft.items():
        excess = float(x[up] - x[dn])
        if abs(excess) > 1e-9:
            relaxations.append({"constraint": label, "group": None, "slack": excess, "reason": "soft constraint violated at its penalty"})
    comps = {"alpha": float(alpha @ w), "cost": float(unit_cost @ (x[b] + x[s]))}
    if track_cols:
        comps["tracking"] = float(sum(x[u] * wgt for u, _, wgt in track_cols))
        for f in {f for _, f, _ in track_cols}:
            comps[f"tracking_{f}"] = float(sum(x[u] for u, ff, _ in track_cols if ff == f))
    if l1_cols is not None:
        comps["active_l1"] = float(x[l1_cols].sum())
    if cvar_cols is not None:
        comps["cvar"] = float(x[cvar_cols[0]] + x[cvar_cols[1]].sum() * cvar_cols[2])
    if risk_info is not None:
        comps["active_variance"] = _variance(objective.risk, ids, w - (wb if objective.risk.active else 0.0))
    return OptimisationResult(weights, True, status, obj, comps, binding, dual_map, relaxations, turnover_all(cs.current, weights), int((np.abs(w) > settings.zero_tol).sum()), solve_s, problem)


def _add_cvar(M: _Model, cs: CrossSection, b, s, w0, wb, n, beta: float, active: bool, constraint: ConstraintSpec | None = None, weight: float = 1.0):
    if cs.scenarios is None:
        raise OptimisationError("CVaR needs cs.scenarios (scenario x id returns)")
    R = cs.scenarios.reindex(columns=cs.ids).fillna(0.0).to_numpy(dtype=float)
    S = R.shape[0]
    base = w0 - (wb if active else 0.0)
    coef = 1.0 / ((1.0 - beta) * S)
    zeta = M.add_vars(1, -INF, INF, 0.0 if constraint is not None else weight)[0]
    psi = M.add_vars(S, 0.0, INF, 0.0 if constraint is not None else weight * coef)
    rows = np.repeat(np.arange(S), 2 * n + 2)
    cols = np.concatenate([np.concatenate([[psi[k], zeta], b, s]) for k in range(S)])
    vals = np.concatenate([np.concatenate([[1.0, 1.0], R[k], -R[k]]) for k in range(S)])
    M.add_rows(rows, cols, vals, -(R @ base), np.full(S, np.inf), [f"cvar:{k}" for k in range(S)])
    if constraint is not None:
        row = M.add_row(np.concatenate([[zeta], psi]), np.concatenate([[1.0], np.full(S, coef)]), -INF, constraint.max, constraint.label)
        if not constraint.hard:
            M.soften(row, constraint.penalty, constraint.label)
    return (zeta, psi, coef)


def _add_risk(M: _Model, rm: RiskModel, ids, b, s, w0, wb, n):
    L = rm.loadings.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    F = rm.factor_cov.to_numpy(dtype=float)
    spec = rm.specific_var.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    k = F.shape[0]
    d = w0 - (wb if rm.active else 0.0)
    lam = rm.aversion
    y = M.add_vars(k, -INF, INF)
    # y - L'(b - s) = L'd
    for j in range(k):
        M.add_row(np.concatenate([[y[j]], b, s]), np.concatenate([[1.0], -L[:, j], L[:, j]]), float(L[:, j] @ d), float(L[:, j] @ d), f"factor:{rm.factor_cov.columns[j]}")
    for i in range(k):
        for j in range(i + 1):
            v = 2.0 * lam * F[i, j]
            if v:
                M.add_q(y[i], y[j], v)
    for i in range(n):
        if spec[i] <= 0:
            continue
        v = 2.0 * lam * spec[i]
        M.add_q(b[i], b[i], v)
        M.add_q(s[i], s[i], v)
        M.add_q(s[i], b[i], -v)
    M.cost[0] = M.cost[0] + 2.0 * lam * spec * d
    M.cost[1] = M.cost[1] - 2.0 * lam * spec * d
    M.offset += float(lam * (spec * d * d).sum())
    return y


def _variance(rm: RiskModel, ids, a: np.ndarray) -> float:
    L = rm.loadings.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    F = rm.factor_cov.to_numpy(dtype=float)
    spec = rm.specific_var.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    y = L.T @ a
    return float(y @ F @ y + (spec * a * a).sum())


__all__ = ["CVaRTerm", "Objective", "OptimisationError", "OptimisationResult", "OptimiserSettings", "RiskModel", "TrackingTerm", "optimise"]
