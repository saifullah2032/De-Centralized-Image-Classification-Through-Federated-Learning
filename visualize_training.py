"""
Training Visualization Script
Displays training progress with charts and metrics
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

print("=" * 70)
print("FEDERATED LEARNING TRAINING VISUALIZATION")
print("=" * 70)
print()

# Check if training history exists
history_path = Path("models/model_history.json")
if not history_path.exists():
    print("[X] No training history found!")
    print("   Run federated training first to generate history.")
    exit(1)

# Load training history
with open(history_path, "r") as f:
    history = json.load(f)

rounds = history["rounds"]
accuracies = [acc * 100 for acc in history["accuracies"]]  # Convert to percentage
losses = history["losses"]
times = history.get("aggregation_times", [0] * len(rounds))

print(f"Training History Loaded:")
print(f"  - Total rounds: {len(rounds)}")
print(f"  - Initial accuracy: {accuracies[0]:.2f}%")
print(f"  - Final accuracy: {accuracies[-1]:.2f}%")
print(f"  - Improvement: {accuracies[-1] - accuracies[0]:+.2f}%")
print()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Federated Learning Training Progress", fontsize=16, fontweight="bold")

# Plot 1: Accuracy over rounds
ax1 = axes[0, 0]
ax1.plot(rounds, accuracies, marker="o", linewidth=2, markersize=8, color="#2ecc71")
ax1.fill_between(rounds, accuracies, alpha=0.3, color="#2ecc71")
ax1.axhline(y=85, color="red", linestyle="--", label="Target (85%)", linewidth=2)
ax1.set_xlabel("Round", fontsize=12)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Global Model Accuracy", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 100])

# Add accuracy values on points
for i, (r, acc) in enumerate(zip(rounds, accuracies)):
    ax1.annotate(
        f"{acc:.1f}%",
        (r, acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )

# Plot 2: Loss over rounds
ax2 = axes[0, 1]
ax2.plot(rounds, losses, marker="s", linewidth=2, markersize=8, color="#e74c3c")
ax2.fill_between(rounds, losses, alpha=0.3, color="#e74c3c")
ax2.set_xlabel("Round", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Global Model Loss", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)

# Add loss values on points
for i, (r, loss) in enumerate(zip(rounds, losses)):
    ax2.annotate(
        f"{loss:.2f}",
        (r, loss),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )

# Plot 3: Accuracy improvement per round
ax3 = axes[1, 0]
acc_improvements = [0] + [
    accuracies[i] - accuracies[i - 1] for i in range(1, len(accuracies))
]
colors = ["green" if x >= 0 else "red" for x in acc_improvements]
ax3.bar(rounds, acc_improvements, color=colors, alpha=0.7, edgecolor="black")
ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
ax3.set_xlabel("Round", fontsize=12)
ax3.set_ylabel("Accuracy Change (%)", fontsize=12)
ax3.set_title("Accuracy Improvement per Round", fontsize=14, fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")

# Add values on bars
for i, (r, imp) in enumerate(zip(rounds, acc_improvements)):
    ax3.annotate(
        f"{imp:+.1f}%",
        (r, imp),
        textcoords="offset points",
        xytext=(0, 5 if imp >= 0 else -15),
        ha="center",
        fontsize=9,
    )

# Plot 4: Training metrics summary
ax4 = axes[1, 1]
ax4.axis("off")

# Create summary table
summary_text = f"""
TRAINING SUMMARY
{"=" * 40}

Rounds Completed:     {len(rounds)}
Duration per Round:   ~{np.mean(times):.1f}s avg

Initial Performance:
  • Accuracy:  {accuracies[0]:.2f}%
  • Loss:      {losses[0]:.3f}

Final Performance:
  • Accuracy:  {accuracies[-1]:.2f}%
  • Loss:      {losses[-1]:.3f}

Overall Improvement:
  • Accuracy:  {accuracies[-1] - accuracies[0]:+.2f}%
  • Loss:      {losses[-1] - losses[0]:+.3f}

Best Round:          Round {rounds[accuracies.index(max(accuracies))]}
Best Accuracy:       {max(accuracies):.2f}%

Target Accuracy:     85.00%
Progress to Target:  {(accuracies[-1] / 85.0) * 100:.1f}%
"""

ax4.text(
    0.1,
    0.5,
    summary_text,
    fontsize=11,
    family="monospace",
    verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

plt.tight_layout()

# Save visualization
viz_path = Path("models/training_visualization.png")
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
print(f"[OK] Visualization saved: {viz_path}")

# Don't display interactively (causes hang on Windows)
print("\n[OK] Visualization saved (interactive display skipped)")
print(f"  Open the saved image: {viz_path}")

print()

# Print detailed metrics table
print("=" * 70)
print("DETAILED METRICS TABLE")
print("=" * 70)
print(f"{'Round':<8} {'Accuracy':<12} {'Loss':<12} {'Acc Change':<15} {'Time (s)':<10}")
print("-" * 70)

for i, r in enumerate(rounds):
    acc_change = acc_improvements[i]
    change_str = f"{acc_change:+.2f}%" if i > 0 else "baseline"
    time_str = f"{times[i]:.2f}" if i < len(times) else "N/A"

    print(
        f"{r:<8} {accuracies[i]:<12.2f} {losses[i]:<12.4f} {change_str:<15} {time_str:<10}"
    )

print("=" * 70)
print()

# Predictions for reaching target
if accuracies[-1] < 85:
    rounds_needed = len(rounds)
    current_rate = (accuracies[-1] - accuracies[0]) / len(rounds)
    if current_rate > 0:
        remaining = 85 - accuracies[-1]
        estimated_rounds = int(remaining / current_rate)
        print(f"[PROJECTIONS]:")
        print(f"  Current improvement rate: {current_rate:.2f}% per round")
        print(
            f"  Estimated rounds to reach 85%: ~{estimated_rounds + len(rounds)} total"
        )
        print(f"  (Need ~{estimated_rounds} more rounds)")
        print()
else:
    print(f"[SUCCESS] TARGET ACHIEVED! Accuracy reached {accuracies[-1]:.2f}%")
    print()

print("=" * 70)
print("[OK] Visualization complete!")
print("=" * 70)
