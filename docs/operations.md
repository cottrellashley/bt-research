# Live proposals and the operational interface

The same strategy a backtest runs becomes orders a desk can execute (§4.14 of the gap analysis).
This page also covers the library side of an operational paper-portfolio service (§6). Tests:
`tests/test_primitives.py::test_413_*`, `test_414_*` and `test_6_*`, and
`tests/test_propose_loaders.py`.

## From a strategy to orders

```python
from nautilus_fi.propose import propose_from_strategy, write_orders, QOE_TSV

p = propose_from_strategy(strategy, source, "2026-10-30", holdings_face, prices, bonds, cash,
                          coverage_fields=("oasd",), min_trade_bps=5, max_position_weight=0.04,
                          cash_target=0.01, cash_bounds=(0.0, 0.03), reported_nav=book_nav)
p.validation          # check, scope, symbol, status, detail, severity
write_orders(p, "orders.tsv", QOE_TSV)        # refuses while a blocker stands
```

`decide_at` runs the strategy's universe, signal, construction, exit policy and overlay sizing
once, from the book's weights at dirty mid. `build_proposal` then sizes the orders:

* **Prices.** Buys are sized and priced at the ask dirty price, sales at the bid.
* **Rounding.** Buys round down, so they never overspend; trims round to the nearest tradable
  amount; full sales are exact. A residual below the minimum piece is closed.
  `round_up_fraction` buys the minimum piece when the target is at least that share of it.
* **Minimums and caps.** A trade below `max(min_trade_bps x NAV, min_trade_face)` is skipped and
  reported. `max_position_weight` caps the targets.
* **Cash.** `cash_target` scales the targets to `1 - cash_target` and reconciles cash after the
  trades to it; `cash_bounds` sets the minimum (a blocker) and maximum (a warning). Cash is also
  checked on each settlement date.
* **Cash flows.** `cash_flow > 0`, an inflow, only buys; `< 0`, an outflow, only sells.
* **Book value.** `reported_nav` must agree with holdings plus cash within `mv_tolerance`.
* **Coverage.** Each of `analytics` / `coverage_fields` must cover `coverage_min` (default 95%) of
  the target book by weight; the signal is always checked. `carry_analytics` fills analytics
  vendors publish after the close from T-1, within one business day, and each carry is recorded
  as a warning.
* **After rounding.** The strategy's own constraints are checked on the post-trade book.

## Severities

Every check has a severity (`severity_of`). `build_proposal(severities={...})` overrides it by
check name or prefix:

| severity | meaning | defaults |
| --- | --- | --- |
| `blocker` | the file must not be sent; `proposal.ok` is False | missing terms or prices, restricted buys, shorts, pieces and increments, cash by settlement date, cash minimum, MV reconciliation, hard limits, futures past first notice or without margin |
| `approval_required` | may be sent once approved | stale prices, soft limits, coverage shortfalls, futures inside the roll window, a decision that kept the book |
| `warning` | for the record | skipped small trades, residual closes, cash above its maximum or off target, carried analytics, futures lot rounding, an off-schedule run |
| `info` / `approved` | passed, a PM's action, an approval given | |

`proposal.blockers` and `proposal.approvals` list them, and the pre-trade report shows them.

## Futures, feedback and files

* **Futures lines.** `add_futures(proposal, target_contracts, futures, prices, current)` adds
  whole contracts per `FutureTerms(symbol, contract_notional, first_notice, last_trade,
  security_id, margin_init)`. `contracts_from_overlays` converts a decision's overlay notionals.
  Readiness checks cover terms, price and its age, the roll window and initial margin against
  cash. Futures lines carry no cash at the trade.
* **PM feedback.** `apply_feedback(proposal, [{"symbol": "B", "action": "reject"}, {"symbol":
  "C", "action": "resize", "face": 50_000}, {"check": "futures_roll", "action": "approve"}])`
  re-runs the proposal from its inputs:
  * rejected and deferred lines are dropped, and deferrals are listed in `extra["deferred"]`;
  * a resized line is validated at the PM's size;
  * approvals turn the matching `approval_required` checks into `approved`.

  Each action is recorded as a `pm_feedback` check.
* **Files.** A `ColumnMap` carries its header and separator. `QOE_TSV` is the headerless,
  tab-separated QOE layout with fixed columns; confirm it against the import specification. CRD,
  QOE and PORT layouts are as before.

## External books and fills

* `book_to_face(book, factors, unit_size, prices)` reads holdings held in `face`, `notional`
  (original face, times the pool factor), `units` (times a unit size) or `market_value` (divided
  by the dirty price). `lines_in_book_units` writes the orders back in the book's terms.
* `reconcile_fills(proposal, fills, price_tolerance_bp)` compares fills with orders. Each order
  gets one row: ordered, filled, average price, signed slippage in bp, and a status of FILLED,
  PARTIAL, UNFILLED, OVERFILLED, WRONG_SIDE or UNEXPECTED (with PRICE_BREAK).
  `positions_after_fills` gives the book to reconcile against the custodian or PORT.

## The operational interface (§6)

Whether bt-research runs the paper-portfolio service or hands it to `fi-app-paperport` /
`fi-lib-systematicpm` is decision §9.1. Either way the library side is in place:

* **Target book.** `target_book(proposal | decision)` gives one row per instrument in a fixed
  schema (`TARGET_BOOK_COLUMNS`, version `TARGET_BOOK_VERSION`): as-of, portfolio, strategy
  fingerprint, identifiers, current / target / trade face, weights, price and record id.
  `write_target_book` writes it with a `.schema.json` holding the version, the columns and a
  content hash. `read_target_book` refuses another version, other columns or changed content.
* **Book reconciliation.** `reconcile_book(model, actual, gold="actual")` compares by symbol.
  Adjustment rows (`kind == "ADJUSTMENT"`) are left out. Each symbol is MATCH, BREAK or BLOCK; a
  BLOCK is a zero-against-nonzero mismatch, and `attrs["blocked"]` stops an upload. `action` says
  whether to rebase the model to the gold copy or correct the store.
* **Return reconciliation.** `reconcile_returns(internal, external)` reports the error
  statistics in bp against a PREP-style return export.
* **Store protocol.** `BookStore` is what a store implements: `positions`, `cash`, `fills` and
  `save_target`. `FileBookStore(root)` implements it on a directory, with target books
  create-only.
* **Record links.** `Session.promote(result, level, links={...})` links official records to the
  store: `portfolio_id`, `port_id`, `config_version`, `owner`, `signal`, `universe_sql`,
  `attachments` (`governance.attach(path)` pins a file by hash), and so on. The record also keeps
  the strategy's code (`config`) and a `version` per exact name. Records stay create-only in
  the hash chain, and `RecordRegistry.find(port_id=...)` looks them up.

What a service would still add (§6, pending §9.1):

* versioned portfolio configs per period;
* persisted positions, trades and price overrides with evidence;
* restartable incremental runs;
* PORT BBU exports of transactions.
