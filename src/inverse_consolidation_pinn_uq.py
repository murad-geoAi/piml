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


DEFAULT_FIG_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SENSOR_CSV = Path(__file__).resolve().parent / "synthetic_sensor_data.csv"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use("seaborn-v0_8-whitegrid")

FIGURE_BG = "#F4EEE5"
AXES_BG = "#FBF7F1"
TEXT_DARK = "#24303A"
TEXT_MUTED = "#5C6770"
GRID_WARM = "#D7CEC1"
BLUE = "#295C77"
TEAL = "#4F7D73"
GREEN = "#608B55"
ORANGE = "#C8783E"
RUST = "#A6482E"
GOLD = "#D1A54A"

CMAP_FIELD = LinearSegmentedColormap.from_list(
    "field_map",
    [TEAL, GREEN, GOLD, ORANGE, RUST],
)
CMAP_UNCERTAINTY = LinearSegmentedColormap.from_list(
    "uncertainty_map",
    ["#F2EFE8", "#D9CDAA", GOLD, ORANGE, RUST],
)

plt.rcParams.update(
    {
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "figure.facecolor": FIGURE_BG,
        "axes.facecolor": AXES_BG,
        "savefig.facecolor": FIGURE_BG,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": TEXT_DARK,
        "axes.edgecolor": TEXT_DARK,
        "text.color": TEXT_DARK,
        "xtick.color": TEXT_DARK,
        "ytick.color": TEXT_DARK,
        "grid.color": GRID_WARM,
        "grid.linewidth": 0.65,
        "grid.alpha": 0.55,
        "legend.frameon": True,
        "legend.facecolor": "#FFFDF9",
        "legend.edgecolor": "#CFC5B7",
        "legend.framealpha": 0.95,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


ProgressCallback = Callable[[int, int, float, float, float], None]


@dataclass(frozen=True)
class ExperimentConfig:
    depth: float = 2.0
    time_factor_max: float = 1.0
    initial_pressure: float = 1.0
    true_cv: float = 1.2
    init_cv: float = 0.5
    noise_std: float = 0.05
    sensor_count: int = 50
    interior_points: int = 800
    boundary_points: int = 240
    initial_points: int = 240
    epochs: int = 1200
    mc_samples: int = 150
    hidden_layers: int = 5
    hidden_units: int = 48
    dropout_p: float = 0.10
    learning_rate: float = 1e-3
    activation_name: str = "tanh"
    polynomial_degree: int = 8
    slice_time: float = 0.2
    eval_points: int = 121
    scheduler_step_size: int = 300
    scheduler_gamma: float = 0.55
    seed: int = 42
    progress_interval: int = 40


@dataclass
class AnalysisResults:
    config: ExperimentConfig
    device_name: str
    sensor_z: np.ndarray
    sensor_t: np.ndarray
    sensor_u_clean: np.ndarray
    sensor_u_noisy: np.ndarray
    sensor_mean_prediction: np.ndarray
    z_grid: np.ndarray
    t_grid: np.ndarray
    true_grid: np.ndarray
    mean_grid: np.ndarray
    std_grid: np.ndarray
    lower_grid: np.ndarray
    upper_grid: np.ndarray
    error_grid: np.ndarray
    z_slice: np.ndarray
    true_slice: np.ndarray
    poly_slice: np.ndarray
    pinn_mean_slice: np.ndarray
    pinn_lower_slice: np.ndarray
    pinn_upper_slice: np.ndarray
    history_epoch: np.ndarray
    total_loss_history: np.ndarray
    data_loss_history: np.ndarray
    physics_loss_history: np.ndarray
    condition_loss_history: np.ndarray
    c_v_history: np.ndarray
    training_runtime_seconds: float
    total_runtime_seconds: float
    metrics: dict[str, float]


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def analytical_consolidation_solution(
    z: np.ndarray,
    t: np.ndarray,
    c_v: float,
    depth: float,
    initial_pressure: float,
    n_terms: int = 200,
) -> np.ndarray:
    z_arr, t_arr = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(t, dtype=float))
    solution = np.zeros_like(z_arr, dtype=float)

    for m in range(n_terms):
        n = 2 * m + 1
        coefficient = 4.0 * initial_pressure / (n * math.pi)
        spatial = np.sin(n * math.pi * z_arr / depth)
        temporal = np.exp(-c_v * (n * math.pi / depth) ** 2 * t_arr)
        solution += coefficient * spatial * temporal

    return solution


def polynomial_features(
    z: np.ndarray,
    t: np.ndarray,
    degree: int,
    depth: float,
    time_factor_max: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    z_scaled = np.asarray(z, dtype=float) / depth
    t_scaled = np.asarray(t, dtype=float) / time_factor_max
    features = []
    powers: list[tuple[int, int]] = []

    for total_degree in range(degree + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            features.append((z_scaled**i) * (t_scaled**j))
            powers.append((i, j))

    return np.column_stack(features), powers


def evaluate_polynomial_regression(
    z: np.ndarray,
    t: np.ndarray,
    coefficients: np.ndarray,
    powers: list[tuple[int, int]],
    depth: float,
    time_factor_max: float,
) -> np.ndarray:
    z_scaled = np.asarray(z, dtype=float) / depth
    t_scaled = np.asarray(t, dtype=float) / time_factor_max
    features = np.column_stack([(z_scaled**i) * (t_scaled**j) for i, j in powers])
    return features @ coefficients


def build_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    if normalized not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[normalized]()


def format_runtime(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 1.0:
        return f"{secs:.1f} s"
    return f"{int(minutes)} min {secs:.1f} s"


def apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.5)
    ax.tick_params(length=0)


def format_domain_axis(ax: plt.Axes, depth: float, time_factor_max: float) -> None:
    ax.set_xlabel("Time factor, $T_v$")
    ax.set_ylabel("Depth, $z$ (m)")
    ax.set_xlim(0.0, time_factor_max)
    ax.set_ylim(depth, 0.0)
    ax.grid(False)


class ConsolidationPINN(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_features = 2

        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(in_features, config.hidden_units))
            layers.append(build_activation(config.activation_name))
            layers.append(nn.Dropout(p=config.dropout_p))
            in_features = config.hidden_units

        layers.append(nn.Linear(config.hidden_units, 1))
        self.network = nn.Sequential(*layers)
        self.c_v = nn.Parameter(torch.tensor([config.init_cv], dtype=torch.float32))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = inputs[:, :1]
        t = inputs[:, 1:]
        z_scaled = 2.0 * (z / self.config.depth) - 1.0
        t_scaled = 2.0 * (t / self.config.time_factor_max) - 1.0
        scaled_inputs = torch.cat([z_scaled, t_scaled], dim=1)
        return self.network(scaled_inputs)


def sample_interior_points(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    z = torch.rand((config.interior_points, 1), device=device) * config.depth
    t = torch.rand((config.interior_points, 1), device=device) * config.time_factor_max
    return torch.cat([z, t], dim=1).requires_grad_(True)


def sample_boundary_points(config: ExperimentConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand((config.boundary_points, 1), device=device) * config.time_factor_max
    top = torch.cat([torch.zeros_like(t), t], dim=1)
    bottom = torch.cat([torch.full_like(t, config.depth), t], dim=1)
    return top, bottom


def sample_initial_points(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    z = torch.rand((config.initial_points, 1), device=device) * config.depth
    t = torch.zeros((config.initial_points, 1), device=device)
    return torch.cat([z, t], dim=1)


def pde_residual(model: ConsolidationPINN, interior_points: torch.Tensor) -> torch.Tensor:
    prediction = model(interior_points)
    first_gradients = torch.autograd.grad(
        prediction,
        interior_points,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
    )[0]
    u_z = first_gradients[:, :1]
    u_t = first_gradients[:, 1:]
    second_gradients = torch.autograd.grad(
        u_z,
        interior_points,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True,
    )[0]
    u_zz = second_gradients[:, :1]
    return u_t - model.c_v * u_zz


def mc_dropout_prediction(
    model: ConsolidationPINN,
    inputs: torch.Tensor,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.train()
    predictions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            predictions.append(model(inputs).squeeze(-1).cpu().numpy())

    stacked = np.stack(predictions, axis=0)
    mean_prediction = stacked.mean(axis=0)
    std_prediction = stacked.std(axis=0)
    lower = mean_prediction - 2.0 * std_prediction
    upper = mean_prediction + 2.0 * std_prediction
    model.eval()
    return mean_prediction, std_prediction, lower, upper


def train_pinn(
    model: ConsolidationPINN,
    sensor_inputs: torch.Tensor,
    sensor_targets: torch.Tensor,
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    total_loss_history: list[float] = []
    data_loss_history: list[float] = []
    physics_loss_history: list[float] = []
    condition_loss_history: list[float] = []
    c_v_history: list[float] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, config.scheduler_step_size),
        gamma=config.scheduler_gamma,
    )
    criterion = nn.MSELoss()

    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sensor_prediction = model(sensor_inputs)
        data_loss = criterion(sensor_prediction, sensor_targets)

        interior_points = sample_interior_points(config, sensor_inputs.device)
        residual = pde_residual(model, interior_points)
        physics_loss = torch.mean(residual**2)

        top_boundary, bottom_boundary = sample_boundary_points(config, sensor_inputs.device)
        top_loss = criterion(model(top_boundary), torch.zeros((config.boundary_points, 1), device=sensor_inputs.device))
        bottom_loss = criterion(
            model(bottom_boundary),
            torch.zeros((config.boundary_points, 1), device=sensor_inputs.device),
        )

        initial_points = sample_initial_points(config, sensor_inputs.device)
        initial_loss = criterion(
            model(initial_points),
            torch.full((config.initial_points, 1), config.initial_pressure, device=sensor_inputs.device),
        )

        condition_loss = top_loss + bottom_loss + initial_loss
        total_loss = data_loss + physics_loss + condition_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.c_v.clamp_(min=1e-4, max=5.0)

        total_value = float(total_loss.detach().cpu().item())
        data_value = float(data_loss.detach().cpu().item())
        physics_value = float(physics_loss.detach().cpu().item())
        condition_value = float(condition_loss.detach().cpu().item())
        c_v_value = float(model.c_v.detach().cpu().item())

        total_loss_history.append(total_value)
        data_loss_history.append(data_value)
        physics_loss_history.append(physics_value)
        condition_loss_history.append(condition_value)
        c_v_history.append(c_v_value)

        should_report = (
            epoch == 1
            or epoch == config.epochs
            or epoch % max(1, config.progress_interval) == 0
        )
        if progress_callback and should_report:
            progress_callback(epoch, config.epochs, data_value, physics_value, c_v_value)

    total_runtime = time.perf_counter() - start_time
    return (
        np.asarray(total_loss_history),
        np.asarray(data_loss_history),
        np.asarray(physics_loss_history),
        np.asarray(condition_loss_history),
        np.asarray(c_v_history),
        total_runtime,
    )


def run_inverse_analysis(
    config: ExperimentConfig,
    progress_callback: ProgressCallback | None = None,
    device: torch.device | None = None,
) -> AnalysisResults:
    overall_start = time.perf_counter()
    configure_reproducibility(config.seed)

    active_device = device or DEFAULT_DEVICE
    rng = np.random.default_rng(config.seed)

    sensor_z = rng.uniform(0.02 * config.depth, 0.98 * config.depth, config.sensor_count)
    sensor_t = np.sort(rng.beta(1.2, 3.0, config.sensor_count) * config.time_factor_max)
    sensor_u_clean = analytical_consolidation_solution(
        sensor_z,
        sensor_t,
        c_v=config.true_cv,
        depth=config.depth,
        initial_pressure=config.initial_pressure,
    )
    sensor_noise = rng.normal(loc=0.0, scale=config.noise_std, size=config.sensor_count)
    sensor_u_noisy = sensor_u_clean + sensor_noise

    sensor_inputs = torch.tensor(
        np.column_stack([sensor_z, sensor_t]),
        dtype=torch.float32,
        device=active_device,
    )
    sensor_targets = torch.tensor(sensor_u_noisy[:, None], dtype=torch.float32, device=active_device)

    model = ConsolidationPINN(config).to(active_device)
    (
        total_loss_history,
        data_loss_history,
        physics_loss_history,
        condition_loss_history,
        c_v_history,
        training_runtime_seconds,
    ) = train_pinn(
        model,
        sensor_inputs,
        sensor_targets,
        config,
        progress_callback=progress_callback,
    )

    z_eval = np.linspace(0.0, config.depth, config.eval_points)
    t_eval = np.linspace(0.0, config.time_factor_max, config.eval_points)
    z_grid, t_grid = np.meshgrid(z_eval, t_eval, indexing="ij")
    eval_inputs = torch.tensor(
        np.column_stack([z_grid.ravel(), t_grid.ravel()]),
        dtype=torch.float32,
        device=active_device,
    )
    mean_flat, std_flat, lower_flat, upper_flat = mc_dropout_prediction(model, eval_inputs, config.mc_samples)
    mean_grid = mean_flat.reshape(z_grid.shape)
    std_grid = std_flat.reshape(z_grid.shape)
    lower_grid = lower_flat.reshape(z_grid.shape)
    upper_grid = upper_flat.reshape(z_grid.shape)

    true_grid = analytical_consolidation_solution(
        z_grid,
        t_grid,
        c_v=config.true_cv,
        depth=config.depth,
        initial_pressure=config.initial_pressure,
    )
    error_grid = np.abs(mean_grid - true_grid)

    z_slice = np.linspace(0.0, config.depth, 300)
    t_slice = np.full_like(z_slice, config.slice_time)
    slice_inputs = torch.tensor(
        np.column_stack([z_slice, t_slice]),
        dtype=torch.float32,
        device=active_device,
    )
    pinn_mean_slice, _, pinn_lower_slice, pinn_upper_slice = mc_dropout_prediction(
        model,
        slice_inputs,
        config.mc_samples,
    )
    true_slice = analytical_consolidation_solution(
        z_slice,
        t_slice,
        c_v=config.true_cv,
        depth=config.depth,
        initial_pressure=config.initial_pressure,
    )

    train_features, train_powers = polynomial_features(
        sensor_z,
        sensor_t,
        degree=config.polynomial_degree,
        depth=config.depth,
        time_factor_max=config.time_factor_max,
    )
    poly_coefficients, *_ = np.linalg.lstsq(train_features, sensor_u_noisy, rcond=None)
    poly_slice = evaluate_polynomial_regression(
        z_slice,
        t_slice,
        poly_coefficients,
        train_powers,
        depth=config.depth,
        time_factor_max=config.time_factor_max,
    )

    sensor_mean_prediction, _, _, _ = mc_dropout_prediction(model, sensor_inputs, config.mc_samples)

    final_c_v = float(model.c_v.detach().cpu().item())
    c_v_abs_error = abs(final_c_v - config.true_cv)
    c_v_rel_error = 100.0 * c_v_abs_error / max(config.true_cv, 1e-12)
    dense_mse = float(np.mean((mean_grid - true_grid) ** 2))
    dense_rmse = float(np.sqrt(dense_mse))
    dense_mae = float(np.mean(error_grid))
    sensor_fit_rmse = float(np.sqrt(np.mean((sensor_mean_prediction - sensor_u_noisy) ** 2)))
    coverage_95 = float(np.mean((true_grid >= lower_grid) & (true_grid <= upper_grid)) * 100.0)
    mean_uncertainty = float(np.mean(std_grid))
    peak_uncertainty_index = np.unravel_index(np.argmax(std_grid), std_grid.shape)
    peak_uncertainty = float(std_grid[peak_uncertainty_index])
    peak_uncertainty_depth = float(z_grid[peak_uncertainty_index])
    peak_uncertainty_time = float(t_grid[peak_uncertainty_index])
    slice_rmse = float(np.sqrt(np.mean((pinn_mean_slice - true_slice) ** 2)))
    baseline_slice_rmse = float(np.sqrt(np.mean((poly_slice - true_slice) ** 2)))
    total_runtime_seconds = time.perf_counter() - overall_start

    metrics = {
        "estimated_cv": final_c_v,
        "true_cv": config.true_cv,
        "cv_abs_error": c_v_abs_error,
        "cv_rel_error_pct": c_v_rel_error,
        "dense_mse": dense_mse,
        "dense_rmse": dense_rmse,
        "dense_mae": dense_mae,
        "sensor_fit_rmse": sensor_fit_rmse,
        "coverage_95_pct": coverage_95,
        "mean_uncertainty": mean_uncertainty,
        "peak_uncertainty": peak_uncertainty,
        "peak_uncertainty_depth": peak_uncertainty_depth,
        "peak_uncertainty_time": peak_uncertainty_time,
        "slice_rmse": slice_rmse,
        "baseline_slice_rmse": baseline_slice_rmse,
        "final_total_loss": float(total_loss_history[-1]),
        "final_data_loss": float(data_loss_history[-1]),
        "final_physics_loss": float(physics_loss_history[-1]),
        "final_condition_loss": float(condition_loss_history[-1]),
    }

    return AnalysisResults(
        config=config,
        device_name=str(active_device),
        sensor_z=sensor_z,
        sensor_t=sensor_t,
        sensor_u_clean=sensor_u_clean,
        sensor_u_noisy=sensor_u_noisy,
        sensor_mean_prediction=sensor_mean_prediction,
        z_grid=z_grid,
        t_grid=t_grid,
        true_grid=true_grid,
        mean_grid=mean_grid,
        std_grid=std_grid,
        lower_grid=lower_grid,
        upper_grid=upper_grid,
        error_grid=error_grid,
        z_slice=z_slice,
        true_slice=true_slice,
        poly_slice=poly_slice,
        pinn_mean_slice=pinn_mean_slice,
        pinn_lower_slice=pinn_lower_slice,
        pinn_upper_slice=pinn_upper_slice,
        history_epoch=np.arange(1, config.epochs + 1),
        total_loss_history=total_loss_history,
        data_loss_history=data_loss_history,
        physics_loss_history=physics_loss_history,
        condition_loss_history=condition_loss_history,
        c_v_history=c_v_history,
        training_runtime_seconds=training_runtime_seconds,
        total_runtime_seconds=total_runtime_seconds,
        metrics=metrics,
    )


def create_loss_figure(results: AnalysisResults) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(results.history_epoch, results.total_loss_history, color=BLUE, linewidth=2.2, label="Total")
    ax.plot(results.history_epoch, results.data_loss_history, color=ORANGE, linewidth=1.8, label="Data")
    ax.plot(results.history_epoch, results.physics_loss_history, color=GREEN, linewidth=1.8, label="PDE")
    ax.plot(
        results.history_epoch,
        results.condition_loss_history,
        color=RUST,
        linewidth=1.8,
        label="BC + IC",
    )
    ax.set_yscale("log")
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", ncol=2)
    apply_axis_style(ax)
    return fig


def create_cv_history_figure(results: AnalysisResults) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(results.history_epoch, results.c_v_history, color=TEAL, linewidth=2.2, label="Learned $c_v$")
    ax.axhline(
        results.metrics["true_cv"],
        color=RUST,
        linestyle="--",
        linewidth=1.5,
        label="Reference $c_v$",
    )
    ax.scatter(
        [results.history_epoch[-1]],
        [results.c_v_history[-1]],
        color=TEAL,
        s=48,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    ax.set_title("Inverse parameter convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$c_v$ (x $10^{-7}$ m$^2$/s)")
    ax.legend(loc="best")
    apply_axis_style(ax)
    return fig


def create_sensor_map_figure(results: AnalysisResults) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    size_scale = 34 + 220 * np.abs(results.sensor_u_noisy) / (np.max(np.abs(results.sensor_u_noisy)) + 1e-8)
    scatter = ax.scatter(
        results.sensor_t,
        results.sensor_z,
        c=results.sensor_u_noisy,
        s=size_scale,
        cmap=CMAP_FIELD,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.92,
    )
    format_domain_axis(ax, results.config.depth, results.config.time_factor_max)
    ax.set_title("Sparse sensor map")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.96, pad=0.03)
    cbar.set_label("Observed pressure, $u$")
    return fig


def create_field_figure(results: AnalysisResults, field_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    if field_name == "mean":
        values = results.mean_grid
        title = "Predictive mean field"
        cmap = CMAP_FIELD
        colorbar_label = "Predicted pressure, $u$"
    elif field_name == "uncertainty":
        values = results.std_grid
        title = "Predictive uncertainty"
        cmap = CMAP_UNCERTAINTY
        colorbar_label = "Predictive std. dev."
    elif field_name == "error":
        values = results.error_grid
        title = "Absolute error"
        cmap = CMAP_UNCERTAINTY
        colorbar_label = "$|\\hat{u} - u|$"
    else:
        raise ValueError(f"Unknown field: {field_name}")

    levels = np.linspace(float(values.min()), float(values.max()) + 1e-10, 18)
    contour = ax.contourf(results.t_grid, results.z_grid, values, levels=levels, cmap=cmap)
    ax.contour(
        results.t_grid,
        results.z_grid,
        values,
        levels=levels[::2],
        colors="white",
        linewidths=0.45,
        alpha=0.55,
    )
    if field_name in {"mean", "uncertainty"}:
        ax.scatter(
            results.sensor_t,
            results.sensor_z,
            s=14,
            color="#FFF8EF",
            edgecolors=TEXT_DARK,
            linewidths=0.35,
            alpha=0.9,
        )

    format_domain_axis(ax, results.config.depth, results.config.time_factor_max)
    ax.set_title(title)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.96, pad=0.03)
    cbar.set_label(colorbar_label)
    return fig


def create_profile_figure(results: AnalysisResults) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.fill_betweenx(
        results.z_slice,
        results.pinn_lower_slice,
        results.pinn_upper_slice,
        color=ORANGE,
        alpha=0.18,
        linewidth=0.0,
        label="95% interval",
    )
    ax.plot(
        results.true_slice,
        results.z_slice,
        color=TEXT_DARK,
        linewidth=2.6,
        label="Analytical",
        zorder=4,
    )
    ax.plot(
        results.pinn_mean_slice,
        results.z_slice,
        color=BLUE,
        linewidth=2.1,
        linestyle="--",
        label="PINN mean",
        zorder=3,
    )
    ax.plot(
        results.poly_slice,
        results.z_slice,
        color=RUST,
        linewidth=1.8,
        linestyle=":",
        label="Polynomial baseline",
        zorder=2,
    )
    ax.set_title(f"Depth profile at $T_v = {results.config.slice_time:.2f}$")
    ax.set_xlabel("Excess pore pressure, $u$")
    ax.set_ylabel("Depth, $z$ (m)")
    ax.set_xlim(-0.15, 1.15 * results.config.initial_pressure)
    ax.set_ylim(results.config.depth, 0.0)
    ax.legend(loc="lower right")
    apply_axis_style(ax)
    return fig


def create_sensor_fit_figure(results: AnalysisResults) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    low = float(min(results.sensor_u_noisy.min(), results.sensor_mean_prediction.min()))
    high = float(max(results.sensor_u_noisy.max(), results.sensor_mean_prediction.max()))
    ax.scatter(
        results.sensor_u_noisy,
        results.sensor_mean_prediction,
        s=38,
        color=BLUE,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.88,
    )
    ax.plot([low, high], [low, high], color=RUST, linestyle="--", linewidth=1.4)
    ax.set_title("Sensor fit check")
    ax.set_xlabel("Observed pressure")
    ax.set_ylabel("Predicted pressure")
    apply_axis_style(ax)
    return fig


def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_report_figures(results: AnalysisResults, output_dir: Path | None = None) -> dict[str, Path]:
    target_dir = output_dir or DEFAULT_FIG_DIR
    paths = {
        "sensor_map": save_figure(create_sensor_map_figure(results), "figure_1_sparse_sensor_map.png", target_dir),
        "training": save_figure(create_loss_figure(results), "figure_2_training_loss.png", target_dir),
        "cv_history": save_figure(create_cv_history_figure(results), "figure_3_cv_history.png", target_dir),
        "profile": save_figure(create_profile_figure(results), "figure_4_depth_profile.png", target_dir),
        "uncertainty": save_figure(create_field_figure(results, "uncertainty"), "figure_5_uncertainty_map.png", target_dir),
        "error": save_figure(create_field_figure(results, "error"), "figure_6_error_map.png", target_dir),
    }
    return paths


def export_sensor_csv(results: AnalysisResults, output_path: Path | None = None) -> Path:
    path = output_path or DEFAULT_SENSOR_CSV
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Depth_z", "TimeFactor_Tv", "Clean_u", "Noisy_u", "PINN_mean_u"])
        for idx in range(results.sensor_z.size):
            writer.writerow(
                [
                    results.sensor_z[idx],
                    results.sensor_t[idx],
                    results.sensor_u_clean[idx],
                    results.sensor_u_noisy[idx],
                    results.sensor_mean_prediction[idx],
                ]
            )
    return path


def print_cli_summary(results: AnalysisResults, figure_paths: dict[str, Path], csv_path: Path) -> None:
    metrics = results.metrics
    print("Inverse consolidation PINN run complete")
    print(f"Device: {results.device_name}")
    print(f"Estimated c_v: {metrics['estimated_cv']:.4f} x 10^-7 m^2/s")
    print(f"Relative error: {metrics['cv_rel_error_pct']:.2f}%")
    print(f"Dense-grid RMSE: {metrics['dense_rmse']:.5f}")
    print(f"95% coverage: {metrics['coverage_95_pct']:.2f}%")
    print(f"Training runtime: {format_runtime(results.training_runtime_seconds)}")
    print(f"Sensor CSV: {csv_path}")
    print("Figures:")
    for label, path in figure_paths.items():
        print(f"- {label}: {path}")


def preset_defaults(preset_name: str) -> dict[str, float | int | str]:
    presets = {
        "Quick preview": {
            "sensor_count": 40,
            "interior_points": 450,
            "boundary_points": 140,
            "initial_points": 140,
            "epochs": 450,
            "mc_samples": 70,
            "hidden_layers": 4,
            "hidden_units": 40,
            "dropout_p": 0.08,
            "learning_rate": 1e-3,
            "activation_name": "tanh",
        },
        "Balanced": {
            "sensor_count": 50,
            "interior_points": 800,
            "boundary_points": 240,
            "initial_points": 240,
            "epochs": 1200,
            "mc_samples": 150,
            "hidden_layers": 5,
            "hidden_units": 48,
            "dropout_p": 0.10,
            "learning_rate": 1e-3,
            "activation_name": "tanh",
        },
        "Research": {
            "sensor_count": 60,
            "interior_points": 1800,
            "boundary_points": 500,
            "initial_points": 500,
            "epochs": 3200,
            "mc_samples": 350,
            "hidden_layers": 6,
            "hidden_units": 64,
            "dropout_p": 0.12,
            "learning_rate": 7e-4,
            "activation_name": "tanh",
        },
    }
    return presets[preset_name]


def metric_card_markup(label: str, value: str, note: str, tone: str = "default") -> str:
    return f"""<div class="micro-card micro-card--{tone}">
        <div class="micro-label">{label}</div>
        <div class="micro-value">{value}</div>
        <div class="micro-note">{note}</div>
    </div>"""


def info_row_markup(label: str, value: str) -> str:
    return f"""<div class="info-row">
        <span>{label}</span>
        <strong>{value}</strong>
    </div>"""


def render_topbar_markup(preset_name: str, has_results: bool) -> str:
    status_text = "Latest run loaded" if has_results else "Ready for a new run"
    status_badge = "Completed" if has_results else "Ready"
    return f"""
    <div class="topbar-shell">
        <div>
            <div class="topbar-label">Dashboard</div>
            <div class="topbar-title">Inverse Consolidation PINN</div>
            <div class="topbar-subtitle">{status_text}</div>
        </div>
        <div class="topbar-actions">
            <div class="search-chip">Search run outputs</div>
            <div class="action-chip">Preset {preset_name}</div>
            <div class="profile-chip">{status_badge}</div>
        </div>
    </div>
    """


def render_summary_markup(results: AnalysisResults) -> str:
    metrics = results.metrics
    config = results.config
    improvement_factor = metrics["baseline_slice_rmse"] / max(metrics["slice_rmse"], 1e-9)
    stat_cards = "".join(
        [
            metric_card_markup(
                "Relative error",
                f"{metrics['cv_rel_error_pct']:.2f}%",
                f"reference {metrics['true_cv']:.4f}",
                tone="mint",
            ),
            metric_card_markup(
                "Dense RMSE",
                f"{metrics['dense_rmse']:.4f}",
                "full field accuracy",
                tone="default",
            ),
            metric_card_markup(
                "Sensor count",
                f"{config.sensor_count}",
                f"noise {config.noise_std:.2f}",
                tone="default",
            ),
            metric_card_markup(
                "MC samples",
                f"{config.mc_samples}",
                f"{improvement_factor:.1f}x profile gain",
                tone="accent",
            ),
        ]
    )
    snapshot_rows = "".join(
        [
            info_row_markup("Epoch budget", f"{config.epochs:,}"),
            info_row_markup("Interior points", f"{config.interior_points:,}"),
            info_row_markup("Architecture", f"{config.hidden_layers} x {config.hidden_units}"),
            info_row_markup("Activation", config.activation_name.upper()),
            info_row_markup("Dropout", f"{config.dropout_p:.2f}"),
            info_row_markup("Runtime", format_runtime(results.training_runtime_seconds)),
        ]
    )
    return f"""
    <div class="summary-grid">
        <div class="feature-card">
            <div class="feature-top">
                <span class="feature-kicker">Model estimate</span>
                <span class="status-pill">Completed</span>
            </div>
            <div class="feature-title">Estimated coefficient of consolidation</div>
            <div class="feature-amount">{metrics['estimated_cv']:.4f}</div>
            <div class="feature-copy">
                x 10^-7 m^2/s
            </div>
            <div class="feature-meta">
                <div class="feature-meta-item">
                    <span>Reference</span>
                    <strong>{metrics['true_cv']:.4f}</strong>
                </div>
                <div class="feature-meta-item">
                    <span>95% coverage</span>
                    <strong>{metrics['coverage_95_pct']:.1f}%</strong>
                </div>
                <div class="feature-meta-item">
                    <span>Training loss</span>
                    <strong>{metrics['final_total_loss']:.3e}</strong>
                </div>
            </div>
        </div>
        <div class="mini-stack">
            {stat_cards}
        </div>
        <div class="rail-card">
            <div class="rail-heading">Run snapshot</div>
            {snapshot_rows}
        </div>
    </div>
    """


def render_pre_run_markup(config: ExperimentConfig, preset_name: str) -> str:
    plan_cards = "".join(
        [
            metric_card_markup("Preset", preset_name, "interactive balance", tone="mint"),
            metric_card_markup("Sensors", f"{config.sensor_count}", "sparse observations", tone="default"),
            metric_card_markup("Epochs", f"{config.epochs:,}", "training budget", tone="default"),
            metric_card_markup("MC samples", f"{config.mc_samples}", "uncertainty passes", tone="accent"),
        ]
    )
    checklist_rows = "".join(
        [
            info_row_markup("Reference c_v", f"{config.true_cv:.2f}"),
            info_row_markup("Initial guess", f"{config.init_cv:.2f}"),
            info_row_markup("Dropout", f"{config.dropout_p:.2f}"),
            info_row_markup("Activation", config.activation_name.upper()),
            info_row_markup("Depth H", f"{config.depth:.2f} m"),
            info_row_markup("Interior points", f"{config.interior_points:,}"),
        ]
    )
    return f"""
    <div class="summary-grid">
        <div class="feature-card">
            <div class="feature-top">
                <span class="feature-kicker">Run blueprint</span>
                <span class="status-pill status-pill--soft">Ready</span>
            </div>
            <div class="feature-title">Model configuration</div>
            <div class="feature-amount">Inverse PINN</div>
            <div class="feature-meta">
                <div class="feature-meta-item">
                    <span>Output</span>
                    <strong>Dashboard</strong>
                </div>
                <div class="feature-meta-item">
                    <span>Preset</span>
                    <strong>{preset_name}</strong>
                </div>
                <div class="feature-meta-item">
                    <span>Focus</span>
                    <strong>Accuracy + UQ</strong>
                </div>
            </div>
        </div>
        <div class="mini-stack">
            {plan_cards}
        </div>
        <div class="rail-card">
            <div class="rail-heading">Current setup</div>
            {checklist_rows}
        </div>
    </div>
    """


def render_insight_stack_markup(results: AnalysisResults) -> str:
    metrics = results.metrics
    config = results.config
    improvement_factor = metrics["baseline_slice_rmse"] / max(metrics["slice_rmse"], 1e-9)
    setup_rows = "".join(
        [
            info_row_markup("Hidden layers", str(config.hidden_layers)),
            info_row_markup("Units per layer", str(config.hidden_units)),
            info_row_markup("Boundary points", f"{config.boundary_points:,}"),
            info_row_markup("Initial points", f"{config.initial_points:,}"),
        ]
    )
    confidence_rows = "".join(
        [
            info_row_markup("Mean uncertainty", f"{metrics['mean_uncertainty']:.4f}"),
            info_row_markup("Peak uncertainty", f"{metrics['peak_uncertainty']:.4f}"),
            info_row_markup("Hotspot depth", f"{metrics['peak_uncertainty_depth']:.2f} m"),
            info_row_markup("Hotspot T_v", f"{metrics['peak_uncertainty_time']:.2f}"),
            info_row_markup("Profile gain", f"{improvement_factor:.1f}x"),
        ]
    )
    return f"""
    <div class="stack-card">
        <div class="stack-card-title">Model setup</div>
        {setup_rows}
    </div>
    <div class="stack-card">
        <div class="stack-card-title">Confidence check</div>
        {confidence_rows}
    </div>
    """


def app_styles() -> str:
    return """
    <style>
        :root {
            --app-bg: #f4f8f2;
            --sidebar-bg: #eef6e9;
            --card-bg: #fbfdf9;
            --card-border: #d9e8d4;
            --text-main: #1e342f;
            --text-muted: #688078;
            --accent: #1f544a;
            --accent-soft: #dcecdf;
            --accent-mid: #2e6a5f;
            --lime: #b8ef8c;
            --lime-soft: #edf8e2;
            --warm: #7fbf68;
            --success: #2e6a5f;
            --shadow: rgba(31, 84, 74, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(184, 239, 140, 0.28), transparent 26%),
                linear-gradient(180deg, #f9fcf7 0%, var(--app-bg) 52%, #eef5ea 100%);
            color: var(--text-main);
            font-family: "Manrope", "Aptos", "Segoe UI", sans-serif;
        }

        .main .block-container {
            max-width: 1420px;
            padding-top: 1.25rem;
            padding-bottom: 2.2rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f4f9ef 0%, var(--sidebar-bg) 100%);
            border-right: 1px solid rgba(31, 84, 74, 0.08);
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption {
            color: var(--text-main);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stNumberInput > div > div,
        [data-testid="stSidebar"] .stTextInput > div > div,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stSelectbox > div > div {
            border-radius: 16px;
        }

        .sidebar-brand {
            background: linear-gradient(160deg, rgba(31, 84, 74, 0.98), rgba(46, 106, 95, 0.94));
            border-radius: 22px;
            padding: 1.1rem 1rem;
            color: #f6fbf7;
            margin-bottom: 1rem;
            box-shadow: 0 18px 30px rgba(31, 84, 74, 0.12);
        }

        .sidebar-brand-tag {
            display: inline-block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            opacity: 0.76;
            margin-bottom: 0.45rem;
        }

        .sidebar-brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.22rem;
        }

        .sidebar-brand-copy {
            font-size: 0.84rem;
            line-height: 1.45;
            color: rgba(246, 251, 247, 0.82);
        }

        .topbar-shell {
            background: rgba(251, 253, 249, 0.88);
            border: 1px solid rgba(31, 84, 74, 0.10);
            border-radius: 24px;
            padding: 1rem 1.15rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            box-shadow: 0 14px 28px var(--shadow);
            margin-bottom: 1rem;
            backdrop-filter: blur(8px);
        }

        .topbar-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.15rem;
        }

        .topbar-title {
            font-size: 1.42rem;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1.15;
        }

        .topbar-subtitle {
            margin-top: 0.18rem;
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .topbar-actions {
            display: grid;
            grid-template-columns: repeat(3, auto);
            gap: 0.55rem;
            align-items: center;
        }

        .search-chip,
        .action-chip,
        .profile-chip {
            border-radius: 999px;
            padding: 0.62rem 0.92rem;
            font-size: 0.84rem;
            white-space: nowrap;
        }

        .search-chip {
            min-width: 14rem;
            background: #ffffff;
            border: 1px solid rgba(31, 84, 74, 0.10);
            color: #91a39e;
        }

        .action-chip {
            background: var(--lime-soft);
            border: 1px solid rgba(184, 239, 140, 0.7);
            color: var(--accent);
            font-weight: 600;
        }

        .profile-chip {
            background: var(--accent);
            color: #f5fbf6;
            font-weight: 700;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: 1.45fr 1fr 0.95fr;
            gap: 0.95rem;
            margin-bottom: 1rem;
        }

        .feature-card {
            background: linear-gradient(160deg, rgba(31, 84, 74, 0.98), rgba(46, 106, 95, 0.94));
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 20px 34px rgba(31, 84, 74, 0.14);
            color: #f7fbf8;
            min-height: 16rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .feature-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
        }

        .feature-kicker {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(247, 251, 248, 0.74);
        }

        .status-pill {
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            background: rgba(184, 239, 140, 0.18);
            border: 1px solid rgba(184, 239, 140, 0.30);
            color: #ebffe0;
            font-size: 0.76rem;
            font-weight: 700;
        }

        .status-pill--soft {
            color: #f7fbf8;
            background: rgba(255, 255, 255, 0.10);
            border-color: rgba(255, 255, 255, 0.16);
        }

        .feature-title {
            margin-top: 1rem;
            font-size: 1rem;
            color: rgba(247, 251, 248, 0.86);
        }

        .feature-amount {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.05;
            margin-top: 0.22rem;
            letter-spacing: -0.02em;
        }

        .feature-copy {
            margin-top: 0.55rem;
            color: rgba(247, 251, 248, 0.78);
            font-size: 0.92rem;
            line-height: 1.5;
            max-width: 34rem;
        }

        .feature-meta {
            margin-top: 1rem;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .feature-meta-item {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            padding: 0.72rem 0.78rem;
        }

        .feature-meta-item span {
            display: block;
            font-size: 0.72rem;
            color: rgba(247, 251, 248, 0.70);
            margin-bottom: 0.2rem;
        }

        .feature-meta-item strong {
            font-size: 1rem;
            color: #f7fbf8;
        }

        .mini-stack {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .micro-card,
        .rail-card,
        .panel-card,
        .stack-card {
            background: rgba(251, 253, 249, 0.95);
            border: 1px solid var(--card-border);
            border-radius: 22px;
            box-shadow: 0 14px 28px var(--shadow);
        }

        .micro-card {
            padding: 1rem;
            min-height: 7.45rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .micro-card--mint {
            background: linear-gradient(180deg, #f4faee 0%, #edf8e2 100%);
        }

        .micro-card--accent {
            background: linear-gradient(180deg, #f7fcf5 0%, #eef7ec 100%);
            border-color: rgba(31, 84, 74, 0.16);
        }

        .micro-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .micro-value {
            font-size: 1.48rem;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1.1;
            margin: 0.24rem 0;
        }

        .micro-note {
            font-size: 0.82rem;
            color: var(--text-muted);
        }

        .rail-card {
            padding: 1rem 1rem 0.9rem 1rem;
        }

        .rail-heading {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .rail-subtitle {
            color: var(--text-muted);
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 0.22rem;
            margin-bottom: 0.7rem;
        }

        .info-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.64rem 0;
            border-top: 1px solid rgba(31, 84, 74, 0.08);
            font-size: 0.88rem;
        }

        .info-row:first-of-type {
            border-top: none;
            padding-top: 0.2rem;
        }

        .info-row span {
            color: var(--text-muted);
        }

        .info-row strong {
            color: var(--text-main);
            font-weight: 700;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.15rem 0 1rem 0;
        }

        .panel-card {
            padding: 0.95rem 1rem;
        }

        .section-kicker {
            color: var(--accent-mid);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-size: 0.72rem;
            margin-bottom: 0.3rem;
        }

        .section-title {
            color: var(--text-main);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: 0;
        }

        .stack-card {
            padding: 0.95rem 1rem;
            margin-bottom: 0.85rem;
        }

        .stack-card-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.25rem;
        }

        .stack-card-copy {
            color: var(--text-muted);
            font-size: 0.84rem;
            line-height: 1.45;
            margin-bottom: 0.45rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(251, 253, 249, 0.80);
            border: 1px solid rgba(31, 84, 74, 0.08);
            min-height: 42px;
            padding: 0.3rem 1rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: var(--lime-soft);
            border-color: rgba(184, 239, 140, 0.92);
            color: var(--accent);
        }

        .stButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            min-height: 2.85rem;
            font-weight: 600;
            border: 1px solid rgba(31, 84, 74, 0.12);
        }

        .stFormSubmitButton button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), var(--accent-mid));
            color: #f5fbf6;
            border: none;
            box-shadow: 0 12px 24px rgba(31, 84, 74, 0.16);
        }

        .stButton > button {
            background: rgba(251, 253, 249, 0.92);
            color: var(--text-main);
        }

        [data-testid="stProgressBar"] > div > div > div {
            background: linear-gradient(90deg, #8adf70 0%, #c0f08b 100%);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--card-border);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 14px 28px var(--shadow);
        }

        @media (max-width: 1100px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
            .feature-meta {
                grid-template-columns: 1fr;
            }
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .topbar-shell {
                flex-direction: column;
                align-items: flex-start;
            }
            .topbar-actions {
                width: 100%;
                grid-template-columns: 1fr;
            }
            .search-chip {
                min-width: 0;
                width: 100%;
            }
        }

        @media (max-width: 760px) {
            .mini-stack {
                grid-template-columns: 1fr;
            }
            .metric-strip {
                grid-template-columns: minmax(0, 1fr);
            }
            .feature-amount {
                font-size: 1.8rem;
            }
        }
    </style>
    """


def render_metrics(results: AnalysisResults) -> str:
    metrics = results.metrics
    cards = [
        metric_card_markup(
            "Sensor-fit RMSE",
            f"{metrics['sensor_fit_rmse']:.4f}",
            "point-wise fit quality",
            tone="default",
        ),
        metric_card_markup(
            "Dense MAE",
            f"{metrics['dense_mae']:.4f}",
            "mean absolute field error",
            tone="default",
        ),
        metric_card_markup(
            "Mean uncertainty",
            f"{metrics['mean_uncertainty']:.4f}",
            "average predictive std. dev.",
            tone="mint",
        ),
        metric_card_markup(
            "Total runtime",
            format_runtime(results.total_runtime_seconds),
            results.device_name,
            tone="accent",
        ),
    ]
    return '<div class="metric-strip">' + "".join(cards) + "</div>"


def build_app_config(st, preset_name: str) -> ExperimentConfig:
    defaults = preset_defaults(preset_name)

    with st.sidebar.form("analysis_controls"):
        st.markdown("### Run setup")
        depth = st.number_input("Layer depth H (m)", min_value=0.5, value=2.0, step=0.1)
        true_cv = st.number_input(
            "Reference c_v (x 10^-7 m^2/s)",
            min_value=0.1,
            value=1.20,
            step=0.05,
            format="%.2f",
        )
        init_cv = st.number_input(
            "Initial c_v guess (x 10^-7 m^2/s)",
            min_value=0.1,
            value=0.50,
            step=0.05,
            format="%.2f",
        )
        noise_std = st.slider("Noise level", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
        sensor_count = st.slider(
            "Sensor count",
            min_value=20,
            max_value=120,
            value=int(defaults["sensor_count"]),
            step=5,
        )

        st.markdown("### Training budget")
        epochs = st.number_input("Epochs", min_value=100, value=int(defaults["epochs"]), step=100)
        interior_points = st.number_input(
            "Interior collocation points",
            min_value=100,
            value=int(defaults["interior_points"]),
            step=50,
        )
        mc_samples = st.number_input(
            "MC dropout samples",
            min_value=20,
            value=int(defaults["mc_samples"]),
            step=10,
        )
        dropout_p = st.slider(
            "Dropout rate",
            min_value=0.0,
            max_value=0.40,
            value=float(defaults["dropout_p"]),
            step=0.01,
        )

        with st.expander("Model details", expanded=False):
            activation_label = st.selectbox(
                "Activation",
                options=["Tanh", "SiLU", "ReLU"],
                index=["tanh", "silu", "relu"].index(str(defaults["activation_name"])),
            )
            hidden_layers = st.number_input(
                "Hidden layers",
                min_value=2,
                max_value=8,
                value=int(defaults["hidden_layers"]),
                step=1,
            )
            hidden_units = st.number_input(
                "Units per layer",
                min_value=16,
                max_value=128,
                value=int(defaults["hidden_units"]),
                step=8,
            )
            learning_rate = st.selectbox("Learning rate", options=[1e-3, 7e-4, 5e-4, 1e-4], index=0)
            slice_time = st.slider(
                "Profile slice T_v",
                min_value=0.05,
                max_value=0.95,
                value=0.20,
                step=0.05,
            )

        run_clicked = st.form_submit_button("Run analysis", use_container_width=True, type="primary")

    if st.sidebar.button("Clear results", use_container_width=True):
        st.session_state.pop("results", None)
        st.rerun()

    return ExperimentConfig(
        depth=depth,
        true_cv=true_cv,
        init_cv=init_cv,
        noise_std=noise_std,
        sensor_count=int(sensor_count),
        interior_points=int(interior_points),
        boundary_points=max(80, int(interior_points) // 3),
        initial_points=max(80, int(interior_points) // 3),
        epochs=int(epochs),
        mc_samples=int(mc_samples),
        hidden_layers=int(hidden_layers),
        hidden_units=int(hidden_units),
        dropout_p=dropout_p,
        learning_rate=float(learning_rate),
        activation_name=activation_label.lower(),
        slice_time=float(slice_time),
        scheduler_step_size=max(100, int(epochs) // 4),
        progress_interval=max(10, int(epochs) // 20),
    ), run_clicked


def show_figure(st, fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_dashboard(st, results: AnalysisResults) -> None:
    import pandas as pd

    st.title("Inverse Consolidation PINN Results")
    metrics = results.metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estimated c_v", f"{metrics['estimated_cv']:.4f}", f"{metrics['cv_rel_error_pct']:.2f}% err")
    col2.metric("Dense RMSE", f"{metrics['dense_rmse']:.4f}")
    col3.metric("Sensor-fit RMSE", f"{metrics['sensor_fit_rmse']:.4f}")
    col4.metric("95% Coverage", f"{metrics['coverage_95_pct']:.1f}%")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Optimization trajectory")
        show_figure(st, create_loss_figure(results))
    with col2:
        st.subheader("Observed vs predicted")
        show_figure(st, create_sensor_fit_figure(results))

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Depth-wise response")
        show_figure(st, create_profile_figure(results))
    with col2:
        st.subheader("c_v convergence")
        show_figure(st, create_cv_history_figure(results))

    tabs = st.tabs(["Field maps", "Sensors", "Run details"])

    with tabs[0]:
        top_left, top_right = st.columns(2, gap="large")
        with top_left:
            show_figure(st, create_field_figure(results, "mean"))
        with top_right:
            show_figure(st, create_field_figure(results, "uncertainty"))
        show_figure(st, create_field_figure(results, "error"))

    with tabs[1]:
        sensor_col, table_col = st.columns([1.08, 0.92], gap="large")
        with sensor_col:
            show_figure(st, create_sensor_map_figure(results))
        with table_col:
            sensor_frame = pd.DataFrame(
                {
                    "z (m)": results.sensor_z,
                    "T_v": results.sensor_t,
                    "Observed u": results.sensor_u_noisy,
                    "PINN mean u": results.sensor_mean_prediction,
                }
            ).sort_values(by="T_v")
            st.dataframe(sensor_frame.round(4), use_container_width=True, height=430)

    with tabs[2]:
        metrics_frame = pd.DataFrame(
            {
                "Metric": [
                    "Estimated c_v",
                    "Reference c_v",
                    "Relative error (%)",
                    "Dense RMSE",
                    "Dense MAE",
                    "Sensor-fit RMSE",
                    "95% coverage (%)",
                    "Mean uncertainty",
                    "Peak uncertainty",
                    "Peak uncertainty depth (m)",
                    "Peak uncertainty T_v",
                    "Profile RMSE",
                    "Polynomial profile RMSE",
                    "Final total loss",
                    "Final data loss",
                    "Final PDE loss",
                    "Final BC+IC loss",
                    "Training runtime (s)",
                    "Total runtime (s)",
                ],
                "Value": [
                    results.metrics["estimated_cv"],
                    results.metrics["true_cv"],
                    results.metrics["cv_rel_error_pct"],
                    results.metrics["dense_rmse"],
                    results.metrics["dense_mae"],
                    results.metrics["sensor_fit_rmse"],
                    results.metrics["coverage_95_pct"],
                    results.metrics["mean_uncertainty"],
                    results.metrics["peak_uncertainty"],
                    results.metrics["peak_uncertainty_depth"],
                    results.metrics["peak_uncertainty_time"],
                    results.metrics["slice_rmse"],
                    results.metrics["baseline_slice_rmse"],
                    results.metrics["final_total_loss"],
                    results.metrics["final_data_loss"],
                    results.metrics["final_physics_loss"],
                    results.metrics["final_condition_loss"],
                    results.training_runtime_seconds,
                    results.total_runtime_seconds,
                ],
            }
        )
        st.dataframe(metrics_frame.round(6), use_container_width=True, height=520)


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Inverse Consolidation PINN",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:
        st.title("Consolidation Lab")
        preset_name = st.selectbox("Preset", options=["Balanced", "Quick preview", "Research"], index=0)

    config, run_clicked = build_app_config(st, preset_name)

    has_results = "results" in st.session_state

    if not has_results and not run_clicked:
        st.info("Configure your run in the sidebar and click 'Run analysis'.")

    if run_clicked:
        progress = st.progress(0, text="Initializing run")
        status = st.empty()

        def update_progress(epoch: int, total_epochs: int, data_loss: float, physics_loss: float, c_v: float) -> None:
            progress.progress(
                int(100 * epoch / total_epochs),
                text=f"Training epoch {epoch}/{total_epochs}",
            )
            status.caption(
                f"Current data loss {data_loss:.3e} | PDE loss {physics_loss:.3e} | c_v {c_v:.4f}"
            )

        results = run_inverse_analysis(config, progress_callback=update_progress)
        st.session_state["results"] = results
        progress.progress(100, text="Run complete")
        time.sleep(0.15)
        progress.empty()
        status.empty()
        st.rerun()

    results = st.session_state.get("results")
    if results is None:
        return

    render_dashboard(st, results)


def main() -> None:
    config = ExperimentConfig(
        interior_points=2000,
        boundary_points=500,
        initial_points=500,
        epochs=5000,
        mc_samples=1000,
        hidden_layers=5,
        hidden_units=50,
        dropout_p=0.10,
        scheduler_step_size=1000,
        scheduler_gamma=0.5,
        progress_interval=500,
    )
    results = run_inverse_analysis(config)
    figure_paths = save_report_figures(results)
    csv_path = export_sensor_csv(results)
    print_cli_summary(results, figure_paths, csv_path)


def is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_app()
    else:
        main()
