from __future__ import annotations

try:
    from inverse_consolidation_pinn_uq import run_streamlit_app
except ModuleNotFoundError:
    from piml.inverse_consolidation_pinn_uq import run_streamlit_app


run_streamlit_app()
