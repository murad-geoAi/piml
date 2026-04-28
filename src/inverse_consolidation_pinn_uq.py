"""
Inverse Consolidation PINN — redesigned Streamlit dashboard.

Design direction: "Deep Lab" — dark scientific instrument aesthetic.
  • Near-black backgrounds (#0D1117) with steel-blue and amber accents
  • All matplotlib figures share the same dark palette as the UI
  • Clear typographic hierarchy: section kickers → titles → body
  • Animated progress, grouped sidebar controls, metric cards with weight
  • Cohesive accent colour: #4FC3F7 (cool cyan) + #FFB74D (warm amber)

Run:
    streamlit run inverse_consolidation_pinn.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────────
DEFAULT_FIG_DIR   = Path(__file__).resolve().parent / "figures"
DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SENSOR_CSV = Path(__file__).resolve().parent / "synthetic_sensor_data.csv"
DEFAULT_DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── unified dark colour palette (UI + matplotlib) ──────────────────────────────
BG_DEEP    = "#0D1117"   # page / figure background
BG_PANEL   = "#161B22"   # card / axes background
BG_BORDER  = "#21262D"   # subtle borders
CYAN       = "#4FC3F7"   # primary accent
AMBER      = "#FFB74D"   # secondary accent
GREEN_ACT  = "#56D364"   # success / positive
RED_ERR    = "#F85149"   # error / negative
TEXT_HI    = "#E6EDF3"   # high-emphasis text
TEXT_MID   = "#8B949E"   # mid-emphasis text
TEXT_DIM   = "#484F58"   # low-emphasis / borders

CMAP_FIELD = LinearSegmentedColormap.from_list(
    "field_dark", ["#0D2137", "#1A4D6B", CYAN, "#B2EBF2", "#FFFFFF"]
)
CMAP_ERROR = LinearSegmentedColormap.from_list(
    "error_dark", ["#0D1117", "#3D1F00", AMBER, "#FF8C00", RED_ERR]
)
CMAP_UQ    = LinearSegmentedColormap.from_list(
    "uq_dark",   ["#0D1117", "#1A2744", "#2C5F8A", CYAN, "#B2EBF2"]
)

plt.rcParams.update({
    "figure.dpi"        : 160,
    "savefig.dpi"       : 160,
    "figure.facecolor"  : BG_DEEP,
    "axes.facecolor"    : BG_PANEL,
    "savefig.facecolor" : BG_DEEP,
    "axes.edgecolor"    : BG_BORDER,
    "axes.labelcolor"   : TEXT_MID,
    "axes.titlecolor"   : TEXT_HI,
    "text.color"        : TEXT_HI,
    "xtick.color"       : TEXT_MID,
    "ytick.color"       : TEXT_MID,
    "grid.color"        : BG_BORDER,
    "grid.linewidth"    : 0.6,
    "grid.alpha"        : 0.8,
    "font.family"       : "monospace",
    "font.size"         : 9,
    "axes.titlesize"    : 10,
    "axes.labelsize"    : 9,
    "legend.frameon"    : True,
    "legend.facecolor"  : BG_PANEL,
    "legend.edgecolor"  : BG_BORDER,
    "legend.framealpha" : 0.95,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : True,
    "axes.spines.bottom": True,
})

ProgressCallback = Callable[[int, int, float, float, float], None]


# ── dataclasses ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    depth            : float = 2.0
    time_factor_max  : float = 1.0
    initial_pressure : float = 1.0
    true_cv          : float = 1.2
    init_cv          : float = 0.5
    noise_std        : float = 0.05
    sensor_count     : int   = 50
    interior_points  : int   = 800
    boundary_points  : int   = 240
    initial_points   : int   = 240
    epochs           : int   = 1200
    mc_samples       : int   = 150
    hidden_layers    : int   = 5
    hidden_units     : int   = 48
    dropout_p        : float = 0.10
    learning_rate    : float = 1e-3
    activation_name  : str   = "tanh"
    polynomial_degree: int   = 8
    slice_time       : float = 0.2
    eval_points      : int   = 121
    scheduler_step_size: int = 300
    scheduler_gamma  : float = 0.55
    seed             : int   = 42
    progress_interval: int   = 40


@dataclass
class AnalysisResults:
    config                  : ExperimentConfig
    device_name             : str
    sensor_z                : np.ndarray
    sensor_t                : np.ndarray
    sensor_u_clean          : np.ndarray
    sensor_u_noisy          : np.ndarray
    sensor_mean_prediction  : np.ndarray
    z_grid                  : np.ndarray
    t_grid                  : np.ndarray
    true_grid               : np.ndarray
    mean_grid               : np.ndarray
    std_grid                : np.ndarray
    lower_grid              : np.ndarray
    upper_grid              : np.ndarray
    error_grid              : np.ndarray
    z_slice                 : np.ndarray
    true_slice              : np.ndarray
    poly_slice              : np.ndarray
    pinn_mean_slice         : np.ndarray
    pinn_lower_slice        : np.ndarray
    pinn_upper_slice        : np.ndarray
    history_epoch           : np.ndarray
    total_loss_history      : np.ndarray
    data_loss_history       : np.ndarray
    physics_loss_history    : np.ndarray
    condition_loss_history  : np.ndarray
    c_v_history             : np.ndarray
    training_runtime_seconds: float
    total_runtime_seconds   : float
    metrics                 : dict[str, float]


# ── helpers ────────────────────────────────────────────────────────────────────
def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def analytical_consolidation_solution(
    z: np.ndarray, t: np.ndarray,
    c_v: float, depth: float, initial_pressure: float,
    n_terms: int = 200,
) -> np.ndarray:
    z_arr, t_arr = np.broadcast_arrays(
        np.asarray(z, dtype=float), np.asarray(t, dtype=float)
    )
    solution = np.zeros_like(z_arr, dtype=float)
    for m in range(n_terms):
        n = 2 * m + 1
        solution += (
            4.0 * initial_pressure / (n * math.pi)
            * np.sin(n * math.pi * z_arr / depth)
            * np.exp(-c_v * (n * math.pi / depth) ** 2 * t_arr)
        )
    return solution


def polynomial_features(
    z: np.ndarray, t: np.ndarray, degree: int,
    depth: float, time_factor_max: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    z_s = np.asarray(z, dtype=float) / depth
    t_s = np.asarray(t, dtype=float) / time_factor_max
    features, powers = [], []
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            features.append((z_s ** i) * (t_s ** j))
            powers.append((i, j))
    return np.column_stack(features), powers


def evaluate_polynomial_regression(
    z, t, coefficients, powers, depth, time_factor_max,
) -> np.ndarray:
    z_s = np.asarray(z, dtype=float) / depth
    t_s = np.asarray(t, dtype=float) / time_factor_max
    F   = np.column_stack([(z_s ** i) * (t_s ** j) for i, j in powers])
    return F @ coefficients


def build_activation(name: str) -> nn.Module:
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[name.strip().lower()]()


def format_runtime(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    return f"{int(m)}m {s:.1f}s" if m >= 1 else f"{s:.1f}s"


# ── neural network ─────────────────────────────────────────────────────────────
class ConsolidationPINN(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_f = 2
        for _ in range(config.hidden_layers):
            layers += [nn.Linear(in_f, config.hidden_units),
                       build_activation(config.activation_name),
                       nn.Dropout(p=config.dropout_p)]
            in_f = config.hidden_units
        layers.append(nn.Linear(config.hidden_units, 1))
        self.network = nn.Sequential(*layers)
        self.c_v = nn.Parameter(torch.tensor([config.init_cv], dtype=torch.float32))
        self._initialize()

    def _initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = inputs[:, :1]
        t = inputs[:, 1:]
        z_s = 2.0 * (z / self.config.depth) - 1.0
        t_s = 2.0 * (t / self.config.time_factor_max) - 1.0
        return self.network(torch.cat([z_s, t_s], dim=1))


# ── collocation sampling ───────────────────────────────────────────────────────
def sample_interior_points(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    z = torch.rand((config.interior_points, 1), device=device) * config.depth
    t = torch.rand((config.interior_points, 1), device=device) * config.time_factor_max
    return torch.cat([z, t], dim=1).requires_grad_(True)


def sample_boundary_points(config: ExperimentConfig, device: torch.device):
    t = torch.rand((config.boundary_points, 1), device=device) * config.time_factor_max
    top    = torch.cat([torch.zeros_like(t), t], dim=1)
    bottom = torch.cat([torch.full_like(t, config.depth), t], dim=1)
    return top, bottom


def sample_initial_points(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    z = torch.rand((config.initial_points, 1), device=device) * config.depth
    t = torch.zeros((config.initial_points, 1), device=device)
    return torch.cat([z, t], dim=1)


def pde_residual(model: ConsolidationPINN, pts: torch.Tensor) -> torch.Tensor:
    pred = model(pts)
    g1   = torch.autograd.grad(pred, pts, torch.ones_like(pred), create_graph=True)[0]
    u_z, u_t = g1[:, :1], g1[:, 1:]
    g2   = torch.autograd.grad(u_z, pts, torch.ones_like(u_z), create_graph=True)[0]
    u_zz = g2[:, :1]
    return u_t - model.c_v * u_zz


# ── MC-dropout inference ───────────────────────────────────────────────────────
def mc_dropout_prediction(
    model: ConsolidationPINN, inputs: torch.Tensor, mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            preds.append(model(inputs).squeeze(-1).cpu().numpy())
    stacked = np.stack(preds)
    mu  = stacked.mean(0)
    sig = stacked.std(0)
    model.eval()
    return mu, sig, mu - 2 * sig, mu + 2 * sig


# ── training loop ──────────────────────────────────────────────────────────────
def train_pinn(
    model: ConsolidationPINN,
    sensor_inputs: torch.Tensor, sensor_targets: torch.Tensor,
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple:
    total_h, data_h, phys_h, cond_h, cv_h = [], [], [], [], []
    opt   = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, config.scheduler_step_size), gamma=config.scheduler_gamma
    )
    mse = nn.MSELoss()
    dev = sensor_inputs.device
    t0  = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        data_loss    = mse(model(sensor_inputs), sensor_targets)
        interior     = sample_interior_points(config, dev)
        phys_loss    = torch.mean(pde_residual(model, interior) ** 2)
        top_b, bot_b = sample_boundary_points(config, dev)
        zeros_b      = torch.zeros((config.boundary_points, 1), device=dev)
        init_pts     = sample_initial_points(config, dev)
        cond_loss    = (mse(model(top_b), zeros_b)
                      + mse(model(bot_b), zeros_b)
                      + mse(model(init_pts),
                            torch.full((config.initial_points, 1),
                                       config.initial_pressure, device=dev)))
        loss = data_loss + phys_loss + cond_loss
        loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            model.c_v.clamp_(1e-4, 5.0)

        total_h.append(float(loss)); data_h.append(float(data_loss))
        phys_h.append(float(phys_loss)); cond_h.append(float(cond_loss))
        cv_h.append(float(model.c_v))

        report = (epoch == 1 or epoch == config.epochs
                  or epoch % max(1, config.progress_interval) == 0)
        if progress_callback and report:
            progress_callback(epoch, config.epochs,
                              float(data_loss), float(phys_loss), float(model.c_v))

    return (np.asarray(total_h), np.asarray(data_h),
            np.asarray(phys_h),  np.asarray(cond_h),
            np.asarray(cv_h),    time.perf_counter() - t0)


# ── main analysis pipeline ─────────────────────────────────────────────────────
def run_inverse_analysis(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
    device: torch.device | None = None,
) -> AnalysisResults:
    t0 = time.perf_counter()
    configure_reproducibility(config.seed)
    dev = device or DEFAULT_DEVICE
    rng = np.random.default_rng(config.seed)

    # sensor data
    sz  = rng.uniform(0.02 * config.depth, 0.98 * config.depth, config.sensor_count)
    st_ = np.sort(rng.beta(1.2, 3.0, config.sensor_count) * config.time_factor_max)
    su_clean = analytical_consolidation_solution(sz, st_, config.true_cv,
                                                 config.depth, config.initial_pressure)
    su_noisy = su_clean + rng.normal(0, config.noise_std, config.sensor_count)

    si = torch.tensor(np.column_stack([sz, st_]), dtype=torch.float32, device=dev)
    st = torch.tensor(su_noisy[:, None],           dtype=torch.float32, device=dev)

    model = ConsolidationPINN(config).to(dev)
    (total_h, data_h, phys_h, cond_h, cv_h, train_rt) = train_pinn(
        model, si, st, config, progress_callback
    )

    # dense evaluation grid
    ze = np.linspace(0, config.depth,          config.eval_points)
    te = np.linspace(0, config.time_factor_max, config.eval_points)
    zg, tg = np.meshgrid(ze, te, indexing="ij")
    ei = torch.tensor(np.column_stack([zg.ravel(), tg.ravel()]),
                      dtype=torch.float32, device=dev)
    mu_f, sig_f, lo_f, hi_f = mc_dropout_prediction(model, ei, config.mc_samples)
    mu_g  = mu_f.reshape(zg.shape)
    sig_g = sig_f.reshape(zg.shape)
    lo_g  = lo_f.reshape(zg.shape)
    hi_g  = hi_f.reshape(zg.shape)
    true_g = analytical_consolidation_solution(zg, tg, config.true_cv,
                                               config.depth, config.initial_pressure)
    err_g  = np.abs(mu_g - true_g)

    # depth slice
    zsl = np.linspace(0, config.depth, 300)
    tsl = np.full_like(zsl, config.slice_time)
    sli = torch.tensor(np.column_stack([zsl, tsl]), dtype=torch.float32, device=dev)
    pm, _, pl, pu = mc_dropout_prediction(model, sli, config.mc_samples)
    true_sl = analytical_consolidation_solution(zsl, tsl, config.true_cv,
                                                config.depth, config.initial_pressure)

    # polynomial baseline
    pf, pp = polynomial_features(sz, st_, config.polynomial_degree,
                                 config.depth, config.time_factor_max)
    pc, *_ = np.linalg.lstsq(pf, su_noisy, rcond=None)
    poly_sl = evaluate_polynomial_regression(zsl, tsl, pc, pp,
                                             config.depth, config.time_factor_max)

    # sensor predictions
    s_mu, *_ = mc_dropout_prediction(model, si, config.mc_samples)

    final_cv = float(model.c_v)
    cv_ae    = abs(final_cv - config.true_cv)
    pk_idx   = np.unravel_index(np.argmax(sig_g), sig_g.shape)
    metrics  = {
        "estimated_cv"          : final_cv,
        "true_cv"               : config.true_cv,
        "cv_abs_error"          : cv_ae,
        "cv_rel_error_pct"      : 100 * cv_ae / max(config.true_cv, 1e-12),
        "dense_mse"             : float(np.mean((mu_g - true_g) ** 2)),
        "dense_rmse"            : float(np.sqrt(np.mean((mu_g - true_g) ** 2))),
        "dense_mae"             : float(np.mean(err_g)),
        "sensor_fit_rmse"       : float(np.sqrt(np.mean((s_mu - su_noisy) ** 2))),
        "coverage_95_pct"       : float(np.mean((true_g >= lo_g) & (true_g <= hi_g)) * 100),
        "mean_uncertainty"      : float(np.mean(sig_g)),
        "peak_uncertainty"      : float(sig_g[pk_idx]),
        "peak_uncertainty_depth": float(zg[pk_idx]),
        "peak_uncertainty_time" : float(tg[pk_idx]),
        "slice_rmse"            : float(np.sqrt(np.mean((pm - true_sl) ** 2))),
        "baseline_slice_rmse"   : float(np.sqrt(np.mean((poly_sl - true_sl) ** 2))),
        "final_total_loss"      : float(total_h[-1]),
        "final_data_loss"       : float(data_h[-1]),
        "final_physics_loss"    : float(phys_h[-1]),
        "final_condition_loss"  : float(cond_h[-1]),
    }

    return AnalysisResults(
        config=config, device_name=str(dev),
        sensor_z=sz, sensor_t=st_, sensor_u_clean=su_clean, sensor_u_noisy=su_noisy,
        sensor_mean_prediction=s_mu,
        z_grid=zg, t_grid=tg, true_grid=true_g, mean_grid=mu_g,
        std_grid=sig_g, lower_grid=lo_g, upper_grid=hi_g, error_grid=err_g,
        z_slice=zsl, true_slice=true_sl, poly_slice=poly_sl,
        pinn_mean_slice=pm, pinn_lower_slice=pl, pinn_upper_slice=pu,
        history_epoch=np.arange(1, config.epochs + 1),
        total_loss_history=total_h, data_loss_history=data_h,
        physics_loss_history=phys_h, condition_loss_history=cond_h,
        c_v_history=cv_h,
        training_runtime_seconds=train_rt,
        total_runtime_seconds=time.perf_counter() - t0,
        metrics=metrics,
    )


# ── figure factories (dark theme) ──────────────────────────────────────────────
def _fig(w=7, h=4.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    fig.patch.set_facecolor(BG_DEEP)
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_color(BG_BORDER)
    ax.tick_params(colors=TEXT_MID, length=3)
    ax.grid(True, color=BG_BORDER, linewidth=0.6, alpha=0.9)
    return fig, ax


def fig_loss(r: AnalysisResults) -> plt.Figure:
    fig, ax = _fig()
    ep = r.history_epoch
    ax.semilogy(ep, r.total_loss_history,     color=CYAN,      lw=2.2, label="Total")
    ax.semilogy(ep, r.data_loss_history,      color=AMBER,     lw=1.7, label="Data", ls="--")
    ax.semilogy(ep, r.physics_loss_history,   color=GREEN_ACT, lw=1.7, label="PDE",  ls="--")
    ax.semilogy(ep, r.condition_loss_history, color=RED_ERR,   lw=1.7, label="BC+IC",ls=":")
    ax.set_xlabel("Epoch", color=TEXT_MID)
    ax.set_ylabel("Loss (log)", color=TEXT_MID)
    ax.set_title("Training loss", color=TEXT_HI, fontweight="bold")
    ax.legend(fontsize=8, labelcolor=TEXT_HI)
    return fig


def fig_cv_history(r: AnalysisResults) -> plt.Figure:
    fig, ax = _fig()
    ax.plot(r.history_epoch, r.c_v_history,
            color=CYAN, lw=2.2, label="Learned $c_v$")
    ax.axhline(r.metrics["true_cv"], color=AMBER, ls="--", lw=1.5,
               label=f"True $c_v = {r.metrics['true_cv']:.2f}$")
    ax.scatter([r.history_epoch[-1]], [r.c_v_history[-1]],
               color=CYAN, s=55, edgecolors=BG_DEEP, linewidths=1.2, zorder=6)
    ax.set_xlabel("Epoch", color=TEXT_MID)
    ax.set_ylabel("$c_v$ (×10⁻⁷ m²/s)", color=TEXT_MID)
    ax.set_title("$c_v$ convergence", color=TEXT_HI, fontweight="bold")
    ax.legend(fontsize=8, labelcolor=TEXT_HI)
    return fig


def fig_sensor_fit(r: AnalysisResults) -> plt.Figure:
    fig, ax = _fig(5.8, 4.2)
    lo = min(r.sensor_u_noisy.min(), r.sensor_mean_prediction.min()) - 0.05
    hi = max(r.sensor_u_noisy.max(), r.sensor_mean_prediction.max()) + 0.05
    ax.scatter(r.sensor_u_noisy, r.sensor_mean_prediction,
               s=36, color=CYAN, edgecolors=BG_DEEP, linewidths=0.6, alpha=0.88)
    ax.plot([lo, hi], [lo, hi], color=AMBER, ls="--", lw=1.4, label="1:1 line")
    ax.set_xlabel("Observed pressure", color=TEXT_MID)
    ax.set_ylabel("Predicted pressure", color=TEXT_MID)
    ax.set_title("Sensor fit (observed vs predicted)", color=TEXT_HI, fontweight="bold")
    ax.legend(fontsize=8, labelcolor=TEXT_HI)
    return fig


def fig_sensor_map(r: AnalysisResults) -> plt.Figure:
    fig, ax = _fig(7.2, 4.5)
    sz = 32 + 180 * np.abs(r.sensor_u_noisy) / (np.abs(r.sensor_u_noisy).max() + 1e-8)
    sc = ax.scatter(r.sensor_t, r.sensor_z, c=r.sensor_u_noisy, s=sz,
                    cmap=CMAP_FIELD, edgecolors=BG_BORDER, linewidths=0.5, alpha=0.92)
    ax.set_xlabel("Time factor $T_v$", color=TEXT_MID)
    ax.set_ylabel("Depth $z$ (m)", color=TEXT_MID)
    ax.set_title("Sparse sensor distribution", color=TEXT_HI, fontweight="bold")
    ax.set_xlim(0, r.config.time_factor_max)
    ax.set_ylim(r.config.depth, 0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Observed $u$", color=TEXT_MID)
    cb.ax.yaxis.set_tick_params(color=TEXT_MID)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_MID)
    return fig


def fig_profile(r: AnalysisResults) -> plt.Figure:
    fig, ax = _fig(6.5, 4.8)
    ax.fill_betweenx(r.z_slice, r.pinn_lower_slice, r.pinn_upper_slice,
                     color=CYAN, alpha=0.14, linewidth=0, label="95% CI")
    ax.plot(r.true_slice,      r.z_slice, color=TEXT_HI, lw=2.5, label="Analytical", zorder=4)
    ax.plot(r.pinn_mean_slice, r.z_slice, color=CYAN,   lw=2.0, ls="--", label="PINN mean", zorder=3)
    ax.plot(r.poly_slice,      r.z_slice, color=AMBER,  lw=1.6, ls=":",  label="Poly baseline", zorder=2)
    ax.set_xlabel("Excess pore pressure $u$", color=TEXT_MID)
    ax.set_ylabel("Depth $z$ (m)", color=TEXT_MID)
    ax.set_title(f"Depth profile at $T_v={r.config.slice_time:.2f}$",
                 color=TEXT_HI, fontweight="bold")
    ax.set_ylim(r.config.depth, 0)
    ax.set_xlim(-0.10, 1.1 * r.config.initial_pressure)
    ax.legend(fontsize=8, labelcolor=TEXT_HI, loc="lower right")
    return fig


def _field_fig(r: AnalysisResults, kind: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    fig.patch.set_facecolor(BG_DEEP)
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_color(BG_BORDER)
    ax.tick_params(colors=TEXT_MID, length=3)

    if kind == "mean":
        vals  = r.mean_grid; cmap = CMAP_FIELD; label = "Predicted $u$"
        title = "Predictive mean pressure field"
    elif kind == "uncertainty":
        vals  = r.std_grid; cmap = CMAP_UQ; label = "Std. deviation"
        title = "Predictive uncertainty (MC dropout)"
    else:  # error
        vals  = r.error_grid; cmap = CMAP_ERROR; label = "$|\\hat{u} - u|$"
        title = "Absolute error field"

    lvls = np.linspace(vals.min(), vals.max() + 1e-10, 22)
    cf   = ax.contourf(r.t_grid, r.z_grid, vals, levels=lvls, cmap=cmap)
    ax.contour(r.t_grid, r.z_grid, vals, levels=lvls[::3],
               colors=BG_BORDER, linewidths=0.4, alpha=0.6)

    if kind in {"mean", "uncertainty"}:
        ax.scatter(r.sensor_t, r.sensor_z, s=10, color="#FFFFFF",
                   edgecolors=BG_BORDER, linewidths=0.3, alpha=0.7, zorder=5)

    ax.set_xlabel("Time factor $T_v$", color=TEXT_MID)
    ax.set_ylabel("Depth $z$ (m)", color=TEXT_MID)
    ax.set_xlim(0, r.config.time_factor_max)
    ax.set_ylim(r.config.depth, 0)
    ax.set_title(title, color=TEXT_HI, fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(label, color=TEXT_MID)
    cb.ax.yaxis.set_tick_params(color=TEXT_MID)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_MID)
    return fig


# ── Streamlit helpers ──────────────────────────────────────────────────────────
def show(st, fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def preset_defaults(name: str) -> dict:
    return {
        "Quick preview": dict(
            sensor_count=40, interior_points=450, boundary_points=140,
            initial_points=140, epochs=450, mc_samples=70,
            hidden_layers=4, hidden_units=40, dropout_p=0.08,
            learning_rate=1e-3, activation_name="tanh"),
        "Balanced": dict(
            sensor_count=50, interior_points=800, boundary_points=240,
            initial_points=240, epochs=1200, mc_samples=150,
            hidden_layers=5, hidden_units=48, dropout_p=0.10,
            learning_rate=1e-3, activation_name="tanh"),
        "Research": dict(
            sensor_count=60, interior_points=1800, boundary_points=500,
            initial_points=500, epochs=3200, mc_samples=350,
            hidden_layers=6, hidden_units=64, dropout_p=0.12,
            learning_rate=7e-4, activation_name="tanh"),
    }[name]


# ── CSS injection ──────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

:root {
  --bg-deep:   #0D1117;
  --bg-panel:  #161B22;
  --bg-raised: #1C2128;
  --border:    #21262D;
  --cyan:      #4FC3F7;
  --amber:     #FFB74D;
  --green:     #56D364;
  --red:       #F85149;
  --text-hi:   #E6EDF3;
  --text-mid:  #8B949E;
  --text-dim:  #484F58;
  --r:         16px;
  --mono:      'JetBrains Mono', monospace;
  --syne:      'Syne', sans-serif;
}

/* ── global ─────────────────────────────────── */
.stApp {
  background: var(--bg-deep) !important;
  font-family: var(--mono);
  color: var(--text-hi);
}
.main .block-container {
  max-width: 1380px;
  padding: 1rem 2rem 3rem;
}

/* ── sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-mid) !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] { color: var(--text-dim) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
  background: var(--bg-raised) !important;
  border-color: var(--border) !important;
  color: var(--text-hi) !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] [class*="stSlider"] > div > div > div {
  background: var(--cyan) !important;
}

/* ── header ─────────────────────────────────── */
.lab-header {
  background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D2137 100%);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1.6rem 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.lab-header::before {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 340px; height: 340px;
  background: radial-gradient(circle, rgba(79,195,247,.12) 0%, transparent 68%);
  pointer-events: none;
}
.lab-tag {
  font-size: .68rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--cyan);
  margin-bottom: .45rem;
}
.lab-title {
  font-family: var(--syne);
  font-size: 2rem; font-weight: 800;
  color: var(--text-hi); line-height: 1.1;
}
.lab-sub {
  margin-top: .4rem; font-size: .85rem;
  color: var(--text-mid); max-width: 52ch;
}
.status-row {
  display: flex; gap: .6rem; margin-top: 1rem; flex-wrap: wrap;
}
.pill {
  border-radius: 999px; padding: .28rem .78rem;
  font-size: .76rem; font-weight: 600;
  border: 1px solid;
}
.pill-cyan  { background: rgba(79,195,247,.12); border-color: rgba(79,195,247,.35); color: var(--cyan); }
.pill-amber { background: rgba(255,183,77,.10); border-color: rgba(255,183,77,.30); color: var(--amber); }
.pill-green { background: rgba(86,211,100,.10); border-color: rgba(86,211,100,.30); color: var(--green); }
.pill-dim   { background: rgba(72,79,88,.20);   border-color: var(--border);        color: var(--text-mid); }

/* ── metric cards ───────────────────────────── */
.mcard-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: .9rem; margin-bottom: 1.4rem;
}
.mcard {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1rem 1.1rem;
  position: relative; overflow: hidden;
}
.mcard::after {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--accent-bar, var(--cyan));
}
.mcard-label {
  font-size: .68rem; letter-spacing: .10em;
  text-transform: uppercase; color: var(--text-mid);
  margin-bottom: .45rem;
}
.mcard-value {
  font-family: var(--syne);
  font-size: 1.65rem; font-weight: 700;
  color: var(--text-hi); line-height: 1.1;
}
.mcard-sub { font-size: .78rem; color: var(--text-mid); margin-top: .25rem; }
.mcard--cyan  { --accent-bar: var(--cyan);  }
.mcard--amber { --accent-bar: var(--amber); }
.mcard--green { --accent-bar: var(--green); }
.mcard--red   { --accent-bar: var(--red);   }

/* ── section header ─────────────────────────── */
.sec-head {
  display: flex; align-items: baseline; gap: .75rem;
  margin: 1.8rem 0 .8rem;
  padding-bottom: .5rem;
  border-bottom: 1px solid var(--border);
}
.sec-num {
  font-size: .68rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--cyan);
  font-weight: 700;
}
.sec-title {
  font-family: var(--syne);
  font-size: 1.05rem; font-weight: 700;
  color: var(--text-hi);
}
.sec-copy { font-size: .82rem; color: var(--text-mid); margin-left: auto; }

/* ── sidebar brand ──────────────────────────── */
.brand {
  background: linear-gradient(160deg, #0D2137 0%, #1A3D5C 100%);
  border: 1px solid rgba(79,195,247,.2);
  border-radius: var(--r); padding: 1rem 1rem .9rem;
  margin-bottom: 1.2rem;
}
.brand-tag {
  font-size: .66rem; letter-spacing: .13em;
  text-transform: uppercase; color: rgba(79,195,247,.7);
  margin-bottom: .3rem;
}
.brand-title {
  font-family: var(--syne); font-size: 1.05rem;
  font-weight: 800; color: var(--text-hi);
  line-height: 1.2;
}
.brand-sub { font-size: .8rem; color: var(--text-mid); margin-top: .25rem; }

/* ── form section labels ────────────────────── */
.form-sec {
  font-size: .72rem; letter-spacing: .10em;
  text-transform: uppercase; color: var(--cyan);
  margin: 1rem 0 .3rem;
}

/* ── tabs ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-panel) !important;
  border-radius: 10px; padding: .25rem;
  border: 1px solid var(--border);
  gap: .25rem;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  background: transparent !important;
  color: var(--text-mid) !important;
  font-family: var(--mono) !important;
  font-size: .82rem !important;
  padding: .4rem 1rem !important;
  min-height: unset !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(79,195,247,.12) !important;
  color: var(--cyan) !important;
}

/* ── progress bar ───────────────────────────── */
[data-testid="stProgressBar"] > div > div { background: var(--bg-panel); border-radius: 8px; }
[data-testid="stProgressBar"] > div > div > div {
  background: linear-gradient(90deg, var(--cyan), #26C6DA);
  border-radius: 8px;
}

/* ── buttons ────────────────────────────────── */
.stButton > button, .stFormSubmitButton > button {
  font-family: var(--mono) !important;
  border-radius: 8px !important;
  font-size: .84rem !important;
}
.stFormSubmitButton button[kind="primary"] {
  background: linear-gradient(135deg, #1A4D6B 0%, #0D3349 100%) !important;
  border: 1px solid rgba(79,195,247,.35) !important;
  color: var(--cyan) !important;
  font-weight: 700 !important;
  box-shadow: 0 0 18px rgba(79,195,247,.10) !important;
}

/* ── dataframe ──────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  overflow: hidden;
}

/* ── pyplot ─────────────────────────────────── */
[data-testid="stImage"] img,
.stPyplot img { border-radius: 12px; }

/* ── divider ────────────────────────────────── */
hr { border-color: var(--border) !important; margin: .5rem 0 !important; }

@media (max-width: 900px) {
  .mcard-row { grid-template-columns: repeat(2, 1fr); }
  .lab-title { font-size: 1.4rem; }
}
</style>
"""


def mc(label, value, sub, tone="cyan") -> str:
    return f"""<div class="mcard mcard--{tone}">
  <div class="mcard-label">{label}</div>
  <div class="mcard-value">{value}</div>
  <div class="mcard-sub">{sub}</div>
</div>"""


def section(num, title, copy="") -> str:
    return f"""<div class="sec-head">
  <span class="sec-num">{num}</span>
  <span class="sec-title">{title}</span>
  {"<span class='sec-copy'>" + copy + "</span>" if copy else ""}
</div>"""


# ── Streamlit app ──────────────────────────────────────────────────────────────
def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Inverse Consolidation PINN",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLES, unsafe_allow_html=True)

    # ── sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="brand">
          <div class="brand-tag">Deep Lab · PINN</div>
          <div class="brand-title">Consolidation<br>Inverse Solver</div>
          <div class="brand-sub">Physics-informed neural net with MC-dropout UQ</div>
        </div>""", unsafe_allow_html=True)

        preset = st.selectbox("Preset", ["Balanced", "Quick preview", "Research"], index=0)
        defs   = preset_defaults(preset)

        with st.form("controls"):
            st.markdown('<div class="form-sec">── Physical parameters</div>',
                        unsafe_allow_html=True)
            depth   = st.number_input("Layer depth H (m)",  0.5, 10.0, 2.0, 0.1)
            true_cv = st.number_input("True c_v (×10⁻⁷)",  0.1, 5.0,  1.20, 0.05, format="%.2f")
            init_cv = st.number_input("Initial c_v guess",  0.1, 5.0,  0.50, 0.05, format="%.2f")
            noise   = st.slider("Sensor noise σ", 0.0, 0.20, 0.05, 0.01)
            n_sens  = st.slider("Sensor count",   20,  120,  int(defs["sensor_count"]), 5)

            st.markdown('<div class="form-sec">── Training budget</div>',
                        unsafe_allow_html=True)
            epochs  = st.number_input("Epochs",          100, 10000, int(defs["epochs"]),        100)
            n_int   = st.number_input("Interior points",  100, 5000,  int(defs["interior_points"]), 50)
            mc_s    = st.number_input("MC dropout samples", 20, 500,  int(defs["mc_samples"]),   10)
            drop    = st.slider("Dropout rate", 0.0, 0.40, float(defs["dropout_p"]), 0.01)

            st.markdown('<div class="form-sec">── Architecture</div>',
                        unsafe_allow_html=True)
            with st.expander("Network details", expanded=False):
                act   = st.selectbox("Activation", ["Tanh", "SiLU", "ReLU"], index=0)
                h_lay = st.number_input("Hidden layers", 2, 8, int(defs["hidden_layers"]), 1)
                h_uni = st.number_input("Units / layer", 16, 128, int(defs["hidden_units"]), 8)
                lr    = st.selectbox("Learning rate", [1e-3, 7e-4, 5e-4, 1e-4], index=0)
                slc   = st.slider("Profile slice T_v", 0.05, 0.95, 0.20, 0.05)

            run_clicked = st.form_submit_button(
                "▶  Run analysis", use_container_width=True, type="primary"
            )

        if st.button("↺  Clear results", use_container_width=True):
            st.session_state.pop("results", None)
            st.rerun()

    # ── build config ─────────────────────────────────────────────────────────
    config = ExperimentConfig(
        depth=depth, true_cv=true_cv, init_cv=init_cv,
        noise_std=noise, sensor_count=int(n_sens),
        interior_points=int(n_int),
        boundary_points=max(80, int(n_int) // 3),
        initial_points =max(80, int(n_int) // 3),
        epochs=int(epochs), mc_samples=int(mc_s),
        hidden_layers=int(h_lay), hidden_units=int(h_uni),
        dropout_p=drop, learning_rate=float(lr),
        activation_name=act.lower(), slice_time=float(slc),
        scheduler_step_size=max(100, int(epochs) // 4),
        progress_interval=max(10, int(epochs) // 20),
    )

    # ── header ───────────────────────────────────────────────────────────────
    has_results = "results" in st.session_state
    status_pill = ('<span class="pill pill-green">● Run complete</span>'
                   if has_results else
                   '<span class="pill pill-dim">○ Ready</span>')
    st.markdown(f"""
    <div class="lab-header">
      <div class="lab-tag">Geotechnical ML · Inverse Problem</div>
      <div class="lab-title">Inverse Consolidation PINN</div>
      <div class="lab-sub">
        Identifies the coefficient of consolidation c_v from sparse pore-pressure
        sensors using a physics-informed neural network with Monte-Carlo dropout
        uncertainty quantification.
      </div>
      <div class="status-row">
        {status_pill}
        <span class="pill pill-cyan">Preset: {preset}</span>
        <span class="pill pill-amber">{config.epochs:,} epochs · {config.sensor_count} sensors</span>
        <span class="pill pill-dim">{config.hidden_layers}×{config.hidden_units} {act}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── training run ──────────────────────────────────────────────────────────
    if not has_results and not run_clicked:
        st.info("Configure parameters in the sidebar and click **▶ Run analysis** to begin.",
                icon="🧬")
        return

    if run_clicked:
        prog  = st.progress(0, text="Initialising…")
        stat  = st.empty()

        def cb(ep, tot, dl, pl, cv):
            prog.progress(int(100 * ep / tot),
                          text=f"Epoch {ep}/{tot}  —  data loss {dl:.3e}  PDE {pl:.3e}  c_v {cv:.4f}")
            stat.caption(f"Data loss: `{dl:.4e}` · PDE: `{pl:.4e}` · c_v: `{cv:.4f}`")

        r = run_inverse_analysis(config, progress_callback=cb)
        st.session_state["results"] = r
        prog.progress(100, text="Complete ✓"); time.sleep(0.2)
        prog.empty(); stat.empty()
        st.rerun()

    r: AnalysisResults = st.session_state["results"]
    m = r.metrics

    # ── 01 · key metrics ──────────────────────────────────────────────────────
    st.markdown(section("01", "Key metrics", "summary of the latest run"),
                unsafe_allow_html=True)

    err_pct = m["cv_rel_error_pct"]
    err_tone = "green" if err_pct < 10 else ("amber" if err_pct < 25 else "red")
    cards = (
        mc("Estimated c_v",  f"{m['estimated_cv']:.4f}", f"true = {m['true_cv']:.4f}", "cyan")
      + mc("Relative error",  f"{err_pct:.2f}%",          "c_v identification", err_tone)
      + mc("Dense RMSE",      f"{m['dense_rmse']:.4f}",   "full-field accuracy", "amber")
      + mc("95% CI coverage", f"{m['coverage_95_pct']:.1f}%", "uncertainty calibration",
           "green" if m["coverage_95_pct"] > 85 else "amber")
    )
    st.markdown(f'<div class="mcard-row">{cards}</div>', unsafe_allow_html=True)

    cards2 = (
        mc("Sensor-fit RMSE", f"{m['sensor_fit_rmse']:.4f}", "point-wise fit", "amber")
      + mc("Dense MAE",        f"{m['dense_mae']:.4f}",      "mean abs error",  "cyan")
      + mc("Mean uncertainty",  f"{m['mean_uncertainty']:.4f}", "avg predictive σ", "amber")
      + mc("Training time",    format_runtime(r.training_runtime_seconds),
           r.device_name, "dim")
    )
    st.markdown(f'<div class="mcard-row">{cards2}</div>', unsafe_allow_html=True)

    # ── 02 · training diagnostics ─────────────────────────────────────────────
    st.markdown(section("02", "Training diagnostics",
                        "loss decomposition & parameter recovery"),
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(st, fig_loss(r))
    with c2:
        show(st, fig_cv_history(r))

    # ── 03 · field reconstruction ─────────────────────────────────────────────
    st.markdown(section("03", "Field reconstruction",
                        "predicted pressure field & uncertainty"),
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(st, _field_fig(r, "mean"))
    with c2:
        show(st, _field_fig(r, "uncertainty"))

    # ── 04 · depth profile ────────────────────────────────────────────────────
    st.markdown(section("04", "Depth profile comparison",
                        f"PINN vs analytical vs polynomial at T_v = {r.config.slice_time:.2f}"),
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 0.9], gap="medium")
    with c1:
        show(st, fig_profile(r))
    with c2:
        show(st, fig_sensor_fit(r))

    # ── 05 · error & sensor maps ──────────────────────────────────────────────
    st.markdown(section("05", "Error & sensor analysis",
                        "absolute error field and sparse observation map"),
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(st, _field_fig(r, "error"))
    with c2:
        show(st, fig_sensor_map(r))

    # ── 06 · tabbed data ──────────────────────────────────────────────────────
    st.markdown(section("06", "Data tables & run details"), unsafe_allow_html=True)

    import pandas as pd
    tab_metrics, tab_sensors = st.tabs(["Full metrics", "Sensor data"])

    with tab_metrics:
        mf = pd.DataFrame({
            "Metric": list(m.keys()),
            "Value":  list(m.values()),
        })
        st.dataframe(mf.round(6), use_container_width=True, height=560)

    with tab_sensors:
        sf = pd.DataFrame({
            "z (m)"       : r.sensor_z,
            "T_v"         : r.sensor_t,
            "Observed u"  : r.sensor_u_noisy,
            "Clean u"     : r.sensor_u_clean,
            "PINN mean u" : r.sensor_mean_prediction,
            "Residual"    : r.sensor_u_noisy - r.sensor_mean_prediction,
        }).sort_values("T_v")
        st.dataframe(sf.round(5), use_container_width=True, height=460)


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    config = ExperimentConfig(
        interior_points=2000, boundary_points=500, initial_points=500,
        epochs=5000, mc_samples=1000, hidden_layers=5, hidden_units=50,
        scheduler_step_size=1000, scheduler_gamma=0.5, progress_interval=500,
    )
    r = run_inverse_analysis(config)
    print(f"Estimated c_v : {r.metrics['estimated_cv']:.4f}")
    print(f"Relative error: {r.metrics['cv_rel_error_pct']:.2f}%")
    print(f"Dense RMSE    : {r.metrics['dense_rmse']:.5f}")
    print(f"95% coverage  : {r.metrics['coverage_95_pct']:.2f}%")
    print(f"Runtime       : {format_runtime(r.training_runtime_seconds)}")


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit():
        run_streamlit_app()
    else:
        main()