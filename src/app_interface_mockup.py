from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from inverse_consolidation_pinn_uq import run_streamlit_app
except ModuleNotFoundError:
    from src.inverse_consolidation_pinn_uq import run_streamlit_app


run_streamlit_app()
