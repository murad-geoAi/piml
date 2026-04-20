import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Paths and Config
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Warm journal palette matching the data figures
FIGURE_BG = "#F3EDE4"
PANEL = "#FBF6EF"
TEXT_COLOR = "#2F2F2F"
TEXT_MUTED = "#5E5B55"
RED = "#D50000"
SALMON = "#DD5D61"
ORANGE = "#F0A65D"
YELLOW = "#F0C55B"
GREEN = "#8FB85C"
DARK_GREEN = "#5F9B61"
REFERENCE_BLUE = "#6EA0D6"
COLOR_PINN = RED
COLOR_PHYSICS = ORANGE
COLOR_CV = GREEN
COLOR_DATA = COLOR_PINN
COLOR_BKG = PANEL
FACE_PINN = "#F8D6D1"
FACE_PHYSICS = "#FBE5C5"
FACE_CV = "#E7EFCF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": FIGURE_BG,
    "savefig.facecolor": FIGURE_BG,
    "text.color": TEXT_COLOR,
})

fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
fig.patch.set_facecolor(FIGURE_BG)
ax.set_facecolor(FIGURE_BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.5)
ax.axis('off')

def add_box(x, y, width, height, text, edgecolor=COLOR_PINN, facecolor=PANEL, textcolor=TEXT_COLOR, fontsize=12, rx=0.1):
    rect = patches.FancyBboxPatch((x, y), width, height,
                                  boxstyle=f"round,pad={rx}",
                                  edgecolor=edgecolor, facecolor=facecolor, linewidth=2.2, zorder=3)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text,
            horizontalalignment='center', verticalalignment='center',
            fontsize=fontsize, color=textcolor, zorder=4)
    # Return connection points: left, right, top, bottom, center
    return {
        'l': (x, y + height/2),
        'r': (x + width, y + height/2),
        't': (x + width/2, y + height),
        'b': (x + width/2, y),
        'c': (x + width/2, y + height/2)
    }

def add_arrow(p1, p2, color=TEXT_COLOR, lw=2.2):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=2)

def add_arrow_path(points, color=TEXT_COLOR, lw=2.2):
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], color=color, lw=lw, zorder=2)
    p_penult = points[-2]
    p_last = points[-1]
    add_arrow(p_penult, p_last, color=color, lw=lw)

# 1. Inputs
p_input = add_box(0.5, 5.0, 1.2, 1.2, "Inputs\n$(z, t)$", edgecolor=TEXT_COLOR, facecolor=COLOR_BKG)

# 2. Scaling
p_scale = add_box(2.5, 5.0, 1.5, 1.2, "Scaling\n$[-1, 1]$", edgecolor=TEXT_COLOR)

# 3. Neural Network representation
nn_x, nn_y = 4.8, 4.3
nn_w, nn_h = 2.4, 2.6
p_nn = add_box(nn_x, nn_y, nn_w, nn_h, "Neural Network\n\n5 Hidden Layers\n50 Units/Layer\nTanh Activation\nDropout ($p=0.1$)", edgecolor=COLOR_PINN, facecolor=FACE_PINN)

# Let's add some visual "nodes" to the NN box just for aesthetic
for i, x_ratio in enumerate([0.15, 0.5, 0.85]):
    for j, y_ratio in enumerate([0.7, 0.8, 0.9]):
        circle = patches.Circle((nn_x + x_ratio*nn_w, nn_y + y_ratio*nn_h), radius=0.06, color=COLOR_PINN, alpha=0.6, zorder=5)
        ax.add_patch(circle)
        
ax.plot([nn_x+0.15*nn_w, nn_x+0.5*nn_w], [nn_y+0.8*nn_h, nn_y+0.9*nn_h], color=COLOR_PINN, alpha=0.3, zorder=4)
ax.plot([nn_x+0.15*nn_w, nn_x+0.5*nn_w], [nn_y+0.8*nn_h, nn_y+0.7*nn_h], color=COLOR_PINN, alpha=0.3, zorder=4)
ax.plot([nn_x+0.5*nn_w, nn_x+0.85*nn_w], [nn_y+0.8*nn_h, nn_y+0.9*nn_h], color=COLOR_PINN, alpha=0.3, zorder=4)
ax.plot([nn_x+0.5*nn_w, nn_x+0.85*nn_w], [nn_y+0.8*nn_h, nn_y+0.7*nn_h], color=COLOR_PINN, alpha=0.3, zorder=4)

# 4. Output Prediction
p_output = add_box(8.0, 5.0, 1.6, 1.2, r"Predicted\nPore Pressure\n$\hat{u}(z, t)$", edgecolor=COLOR_PINN)

# 5. Data Loss (Top branch)
p_data_loss = add_box(10.2, 5.0, 1.5, 1.2, r"Data Loss\n$\mathcal{L}_{data}$", edgecolor=COLOR_PINN, facecolor=FACE_PINN)

# 6. Automatic Differentiation (Bottom branch)
p_autodiff = add_box(8.0, 2.5, 1.6, 1.2, "Automatic\nDifferentiation\n(Autograd)", edgecolor=COLOR_PHYSICS)

# 7. Discovered Parameter cv
p_cv = add_box(5.0, 2.5, 1.8, 1.2, r"Inverse Parameter\n$c_v$ (Learnable)", edgecolor=COLOR_CV, facecolor=FACE_CV)

# 8. PDE Residual
p_residual = add_box(8.0, 0.4, 1.6, 1.2, r"PDE Residual\n$\mathcal{F}_{PDE}$", edgecolor=COLOR_PHYSICS, facecolor=FACE_PHYSICS)
ax.text(8.8, 0.2, r"$\mathcal{F}_{PDE} = \frac{\partial \hat{u}}{\partial t} - c_v \frac{\partial^2 \hat{u}}{\partial z^2}$", ha='center', va='center', fontsize=11, color=COLOR_PHYSICS)

# 9. Physics Loss
p_physics_loss = add_box(10.2, 0.4, 1.5, 1.2, r"Physics Loss\n$\mathcal{L}_{PDE}$", edgecolor=COLOR_PHYSICS, facecolor=FACE_PHYSICS)

# 10. Total Loss
p_total = add_box(10.2, 2.8, 1.5, 1.2, r"Total Loss\n$\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{PDE}$", edgecolor=TEXT_COLOR, facecolor=COLOR_BKG, fontsize=11)


# Connect Everything using Arrows
add_arrow(p_input['r'], p_scale['l'])
add_arrow(p_scale['r'], (p_nn['l'][0], p_scale['r'][1]))
add_arrow((p_nn['r'][0], p_output['l'][1]), p_output['l'])

# Data loss connection
add_arrow(p_output['r'], p_data_loss['l'])

# Extra Text for Data Loss
ax.text(9.0, 6.4, "Sparse Noisy Sensor Data\n$u_{sensor}$", ha='center', va='center', fontsize=10, color=TEXT_COLOR)
add_arrow_path([(9.0, 6.1), (9.0, 5.6), (10.95, 5.6), (10.95, 6.2)]) # Not quite right, let's just make a simple point

# Point from sensor data to Data loss
p_sensor = add_box(10.2, 6.7, 1.5, 0.6, "Sensor Data\n$u_{sensor}$", edgecolor=TEXT_COLOR, fontsize=10)
add_arrow(p_sensor['b'], p_data_loss['t'])

# Autodiff from Input and Output
add_arrow_path([p_output['b'], p_autodiff['t']], color=COLOR_PHYSICS)
add_arrow_path([p_input['b'], (0.9, 3.1), p_autodiff['l']], color=COLOR_PHYSICS) # Input to autodiff

# Autodiff to PDE residual
add_arrow_path([p_autodiff['b'], p_residual['t']], color=COLOR_PHYSICS)
ax.text(8.95, 2.0, r"$\frac{\partial \hat{u}}{\partial t}, \frac{\partial^2 \hat{u}}{\partial z^2}$", ha='left', va='center', fontsize=12, color=COLOR_PHYSICS)

# cv to PDE residual
add_arrow_path([p_cv['b'], (5.9, 1.0), p_residual['l']], color=COLOR_CV)

# Collocation points
p_coll = add_box(5.0, 0.4, 1.8, 0.8, "Collocation Points\n(Interior)", edgecolor=TEXT_COLOR, fontsize=10)
add_arrow(p_coll['r'], p_residual['l'])

# PDE residual to Physics Loss
add_arrow(p_residual['r'], p_physics_loss['l'])

# Loss terms to Total Loss
add_arrow(p_data_loss['b'], p_total['t'])
add_arrow(p_physics_loss['t'], p_total['b'])

# Optimization
p_opt = add_box(4.8, 6.7, 2.4, 0.6, "Adam Optimizer", edgecolor=TEXT_COLOR, facecolor=COLOR_BKG)
add_arrow_path([p_total['l'], (10.95, 3.4), (10.95, 7.0), p_opt['r']], color=TEXT_COLOR)
add_arrow_path([p_opt['b'], (6.0, 5.6), p_nn['t']], color=TEXT_COLOR) # Update NN
add_arrow_path([p_opt['b'], (5.9, 3.8), p_cv['t']], color=COLOR_CV) # Update cv
ax.text(5.5, 6.3, "Update Weights\n& Biases", ha='right', va='center', fontsize=9, color=TEXT_COLOR)
ax.text(5.5, 3.8, "Update $c_v$", ha='right', va='center', fontsize=9, color=COLOR_CV)

output_path = FIG_DIR / "figure_6_pinn_architecture.png"
plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Successfully generated PINN Architecture Diagram at {output_path}")

plt.savefig(FIG_DIR / "figure_6_pinn_architecture.pdf", bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Successfully generated PDF version at {FIG_DIR / 'figure_6_pinn_architecture.pdf'}")
