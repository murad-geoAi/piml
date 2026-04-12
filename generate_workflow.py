import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set up figure
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("#faf0e6")  # Light linen background from image

# Colors from extracted palette
PRIMARY = "#d00000"
SECONDARY = "#e9874e"
ACCENT_1 = "#8fc067"
ACCENT_2 = "#579d52"
TEXT_DL = "#ffffff"

def draw_box(x, y, w, h, title, subtitle, facecolor, edgecolor="#333333"):
    # Box
    box = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=1.5", 
        linewidth=2.0, edgecolor=edgecolor, facecolor=facecolor,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Title text
    ax.text(x + w/2, y + h - 5.5, title, ha="center", va="center", 
            fontsize=13, fontweight="bold", color=TEXT_DL, family="serif")
    
    # Subtitle text
    ax.text(x + w/2, y + h/2 - 2, subtitle, ha="center", va="center", 
            fontsize=10, color=TEXT_DL, linespacing=1.5, family="serif")

def draw_arrow(x1, y1, x2, y2, label=None, label_pt=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.4", 
                                color="#444444", lw=2.5, shrinkA=0, shrinkB=0))
    if label and label_pt:
        ax.text(label_pt[0], label_pt[1], label, ha="center", va="center", 
                fontsize=10, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#faf0e6", ec="none", alpha=0.9))

# Draw the blocks
# 1. Inputs (Left)
# 1a: Training Data
draw_box(5, 65, 23, 18, "Training Data", 
         "50 Sparse & Noisy\nSensor Measurements\n$(z, t, u_{noisy})$", PRIMARY)

# 1b: Collocation Points
draw_box(5, 35, 23, 18, "Physics Constraints", 
         "2,000 Interior\nCollocation Points\nTerzaghi PDE Domain", PRIMARY)

# 2. Network (Center Top)
draw_box(38, 50, 24, 30, "PINN Architecture", 
         "Multilayer Perceptron\n5 Layers $\\times$ 50 Neurons\n\n+ Dropout ($p=0.1$)\n\nLearnable Parameter: $c_v$", SECONDARY)

# 3. Loss & Optimizer (Center Bottom)
draw_box(38, 10, 24, 25, "Optimization", 
         "Loss = MSE$_{data}$ + MSE$_{PDE}$\n\nAdam Optimizer\n5,000 Epochs\nAutomatic Differentiation", ACCENT_1)

# 4. Uncertainty Quantification (Right Top)
draw_box(71, 58, 24, 22, "Uncertainty\nQuantification", 
         "Monte Carlo Dropout\n1,000 Stochastic Passes\nat Inference Time", ACCENT_2)

# 5. Predictions (Right Bottom)
draw_box(71, 23, 24, 22, "Final Outputs", 
         "Robust Discovery of $c_v$\n\nPredictive Mean $\\hat{u}$\n95% Confidence Bounds", ACCENT_2)

# Connect Inputs to Network
draw_arrow(28, 74, 34, 74)         # Data to Network
draw_arrow(34, 74, 34, 65)         # Data path down
draw_arrow(34, 65, 38, 65, "Forward\nPass", (34, 70))

draw_arrow(28, 44, 34, 44)         # Physics to Network
draw_arrow(34, 44, 34, 65)         # Physics path up

# Connect Network to Loss
draw_arrow(50, 50, 50, 35, "Compute\nResiduals", (50, 42.5))

# Feedback loop: Loss back to Network
draw_arrow(43, 35, 43, 50, "Update Weights\n& $c_v$", (43, 42.5))

# Connect Network to UQ
draw_arrow(62, 65, 71, 65, "", None)

# Connect UQ to Outputs
draw_arrow(83, 58, 83, 45, "Aggregate", (83, 51.5))

# Add an overarching title
ax.text(50, 92, "Methodology Workflow: Uncertainty-Aware PINN for Inverse Consolidation", 
        ha="center", va="center", fontsize=16, fontweight="bold", family="serif", color="#222222")

# Save figure
output_path = Path(__file__).resolve().parent / "figures" / "methodology_workflow.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches="tight")
print(f"Methodological workflow figure successfully generated at {output_path}")
