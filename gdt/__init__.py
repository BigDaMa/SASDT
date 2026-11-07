"""GDT experiment toolkit.

The public API intentionally stays small.  Legacy modules in the repository import
from this package so existing experiment scripts keep working while new code has a
single, predictable home.
"""

from gdt.config import Config, ExperimentConfig
from gdt.data import PairDataset

__all__ = ["Config", "ExperimentConfig", "PairDataset"]
