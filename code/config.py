"""Paths for the release. Everything is relative to the repository root; override with env vars if needed.
  CURVES : frozen measured examination curves (data/curves)
  DATA   : derived (compact) dataset files produced by prepare_data.py (data/derived)
  FIGS   : where figure PDFs are written (outputs/figs)
  TABLES : where assembled table text is written (outputs/tables)
Analysis JSONs are written next to the scripts (this directory), as the scripts read each other's outputs by name.
"""
import os
ROOT = os.environ.get("WF_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
CURVES = os.environ.get("WF_CURVES", os.path.join(ROOT, "data", "curves"))
DATA = os.environ.get("WF_DATA", os.path.join(ROOT, "data", "derived"))
FIGS = os.environ.get("WF_FIGS", os.path.join(ROOT, "outputs", "figs"))
TABLES = os.environ.get("WF_TABLES", os.path.join(ROOT, "outputs", "tables"))
for _d in (FIGS, TABLES):
    os.makedirs(_d, exist_ok=True)
