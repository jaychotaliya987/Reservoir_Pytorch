"""
This module contains dataset generation and loading utilities used internally
within the examples provided in this repository.

Note:
    - These datasets are **not intended for general-purpose use**.
    - Users of this library are expected to supply their own datasets
      when applying the models in real-world scenarios.
    - The contents here serve as demonstration tools for example notebooks
      and scripts.

Feel free to explore or adapt for your own purposes, but keep in mind this
module is maintained with internal examples in mind.
"""

from .LorenzAttractor import LorenzAttractor
from .MackeyGlassDataset import MackeyGlassDataset

__all__ = ['LorentzAttractor', 'MackeyGlassDataset']