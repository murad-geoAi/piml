from __future__ import annotations

import sys
import os

# Ensure the src/ directory is on the import path so sibling modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inverse_consolidation_pinn_uq import run_streamlit_app

run_streamlit_app()
