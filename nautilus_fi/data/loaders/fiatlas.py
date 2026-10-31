"""FIAtlas datasets: a catalogue of mappings from FIAtlas tables to the canonical datasets.

FIAtlas sits in Snowflake, so its loader is ``snowflake_loader`` with this catalogue. The table and
column names below are templates: copy ``fiatlas_catalog.example.json``, set ``variables``
(database, schema) and each dataset's source columns to the deployed FIAtlas views, and load it with
``load_catalog``. Everything downstream (point-in-time stores, the environment builder, the panel
engine) only sees the canonical shapes.
"""

from __future__ import annotations

from pathlib import Path

from nautilus_fi.data.loaders.base import DatasetMapping, load_catalog

EXAMPLE_CATALOG = Path(__file__).with_name("fiatlas_catalog.example.json")


def fiatlas_catalog(path: str | Path | None = None) -> dict[str, DatasetMapping]:
    """The deployed catalogue (``path``) or the shipped example (templates, for tests and as a guide)."""
    return load_catalog(path or EXAMPLE_CATALOG)


__all__ = ["EXAMPLE_CATALOG", "fiatlas_catalog"]
