# PINN-UQ: Inverse Consolidation with Physics-Informed Neural Networks

Research repository for the APMCE 2026 paper on **inverse consolidation analysis** using Physics-Informed Neural Networks (PINNs) with Uncertainty Quantification (UQ) via Monte Carlo Dropout.

---

## Repository Structure

```
piml/
├── src/                            # Core source code
│   ├── inverse_consolidation_pinn_uq.py   # Main PINN model + UQ (MC Dropout)
│   └── app_interface_mockup.py            # Streamlit dashboard interface
│
├── scripts/                        # One-off generation & utility scripts
│   ├── generate_data_figure.py     # Generates data distribution figures
│   ├── generate_pinn_diagram.py    # Generates PINN architecture diagram
│   ├── generate_workflow.py        # Generates methodology workflow figure
│   ├── convert_doc.py              # Document conversion utility
│   └── convert_md_to_docx.py      # Markdown → DOCX conversion
│
├── data/                           # Input datasets
│   └── synthetic_sensor_data.csv  # Synthetic sparse sensor data (50 points)
│
├── figures/                        # Output figures (paper-ready)
│   ├── figure_1_sparse_sensor_map.png
│   ├── figure_2_convergence_and_cv.png
│   ├── figure_3_true_vs_traditional_vs_pinn.png
│   ├── figure_4_uncertainty_map.png
│   ├── figure_5_error_map.png
│   ├── figure_6_pinn_architecture.png / .pdf
│   ├── figure_data_distribution.png / .pdf
│   └── methodology_workflow.png
│
├── docs/                           # Manuscripts and documentation
│   ├── APMCE_2026_Manuscript.docx  # Final manuscript
│   ├── APMCE_2026_Manuscript.md    # Manuscript source (Markdown)
│   ├── Research_Abstract.md        # Research abstract
│   └── (archived paper drafts)
│
├── assets/                         # Screenshots and presentation assets
│   ├── clean_dashboard_native.png
│   ├── dashboard_screenshot.png
│   └── final_dashboard_screenshot.png
│
├── logs/                           # Runtime logs and outputs
│   └── (log files)
│
├── README.md
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch numpy pandas matplotlib streamlit scikit-learn
```

### 2. Run the main PINN model
```bash
python src/inverse_consolidation_pinn_uq.py
```

### 3. Launch the Streamlit dashboard
```bash
streamlit run src/app_interface_mockup.py
```

### 4. Regenerate figures
```bash
python scripts/generate_data_figure.py
python scripts/generate_pinn_diagram.py
python scripts/generate_workflow.py
```

---

## Research Overview

| Item | Detail |
|------|--------|
| **Problem** | Inverse identification of consolidation parameters from sparse sensor data |
| **Method** | Physics-Informed Neural Network (PINN) with MC Dropout UQ |
| **Data** | 50 synthetic sensor points + 2000 collocation points |
| **Conference** | APMCE 2026 |

---

## License

See [LICENSE](LICENSE) for details.