"""
VISUALIZATION CODE: Performance Metrics Charts
===============================================

Generates three professional charts:
1. ACCURACY VS. ROUNDS: Shows training progression
2. THE INTELLIGENCE GAP: Raw CNN vs VLM-Corrected Ensemble
3. INFERENCE LATENCY: Processing time per layer

Outputs publication-ready PNG files for presentations and papers.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

# Professional styling
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Georgia", "Times New Roman"]
rcParams["font.size"] = 11
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 14
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.titlesize"] = 16
rcParams["figure.dpi"] = 300

# Color scheme (Deep Ocean theme)
COLOR_CNN = "#0066cc"  # Deep ocean blue
COLOR_SCL = "#ff9800"  # Orange (correction)
COLOR_ENSEMBLE = "#00d4ff"  # Neon cyan (success)
COLOR_ACCENT = "#4caf50"  # Green (positive)
COLOR_GRID = "#e0e0e0"  # Light gray


class MetricsVisualizer:
    def __init__(self):
        """Initialize visualizer with data"""
        with open("models/model_history.json", "r") as f:
            self.history = json.load(f)

        with open("metrics_current_standing.json", "r") as f:
            self.metrics = json.load(f)

    def chart_1_accuracy_vs_rounds(self):
        """Chart 1: Accuracy improvement across training rounds"""
        print("Generating Chart 1: Accuracy vs. Rounds...")

        fig, ax = plt.subplots(figsize=(12, 7))

        rounds = self.history["rounds"]
        accuracies = [acc * 100 for acc in self.history["accuracies"]]

        # Plot line
        ax.plot(
            rounds,
            accuracies,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=COLOR_CNN,
            label="CNN Accuracy (Federated Learning)",
        )

        # Fill area under curve
        ax.fill_between(rounds, 0, accuracies, alpha=0.2, color=COLOR_CNN)

        # Add best accuracy annotation
        best_idx = accuracies.index(max(accuracies))
        best_round = rounds[best_idx]
        best_acc = accuracies[best_idx]
        ax.annotate(
            f"Peak: {best_acc:.2f}%\n(Round {best_round})",
            xy=(best_round, best_acc),
            xytext=(best_round - 3, best_acc + 5),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
        )

        # Styling
        ax.set_xlabel("Training Round", fontsize=13, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            "Stage 1 CNN: Training Progression Over 20 Federated Rounds",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.grid(True, alpha=0.3, linestyle="--", color=COLOR_GRID)
        ax.set_ylim(0, max(accuracies) + 10)
        ax.set_xlim(0, 21)
        ax.legend(loc="upper left", fontsize=11, framealpha=0.95)

        # Add final value box
        final_acc = accuracies[-1]
        textstr = f"Final Accuracy: {final_acc:.2f}%\nImprovement: {final_acc - accuracies[0]:.2f}%"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.98,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()
        plt.savefig("chart_1_accuracy_vs_rounds.png", dpi=300, bbox_inches="tight")
        print("  Saved: chart_1_accuracy_vs_rounds.png")
        plt.close()

    def chart_2_intelligence_gap(self):
        """Chart 2: Raw CNN vs VLM-Corrected Ensemble comparison"""
        print("Generating Chart 2: The Intelligence Gap...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Data extraction
        cnn_accuracy = self.metrics["stage_1_cnn"]["accuracy"] * 100
        scl_correction = self.metrics["stage_1_5_scl"]["additional_gain"] * 100
        scl_accuracy = self.metrics["stage_1_5_scl"]["estimated_corrected"] * 100
        ensemble_accuracy = self.metrics["stage_4_ensemble"]["ensemble_accuracy"] * 100

        # Bar positions
        categories = [
            "Stage 1\nRaw CNN",
            "Stage 1.5\nSCL Correction",
            "Stage 1.5\nPost-SCL",
            "Stage 4\nEnsemble",
        ]
        values = [cnn_accuracy, scl_correction, scl_accuracy, ensemble_accuracy]
        colors = [COLOR_CNN, COLOR_SCL, COLOR_CNN, COLOR_ENSEMBLE]

        x_pos = np.arange(len(categories))
        bars = ax.bar(
            x_pos, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
        )

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Add improvement arrows and annotations
        # Arrow from CNN to SCL
        ax.annotate(
            "",
            xy=(1, values[0] + 2),
            xytext=(0, values[0] + 2),
            arrowprops=dict(arrowstyle="->", lw=2, color=COLOR_ACCENT),
        )
        ax.text(
            0.5,
            values[0] + 4,
            "+20.23%",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

        # Arrow from SCL to Ensemble
        ax.annotate(
            "",
            xy=(3, values[2] + 2),
            xytext=(2, values[2] + 2),
            arrowprops=dict(arrowstyle="->", lw=2, color=COLOR_ACCENT),
        )
        ax.text(
            2.5,
            values[2] + 4,
            "+15.88%",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

        # Styling
        ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            "The Intelligence Gap: Raw CNN vs VLM-Corrected Ensemble\n"
            + "Demonstrates Multi-Layer Architecture Benefits",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, max(values) + 15)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y", color=COLOR_GRID)

        # Add performance zones
        ax.axhspan(0, 30, alpha=0.05, color="red", label="Low Confidence Zone")
        ax.axhspan(30, 60, alpha=0.05, color="yellow", label="Correction Zone")
        ax.axhspan(60, 100, alpha=0.05, color="green", label="High Confidence Zone")
        ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        plt.savefig("chart_2_intelligence_gap.png", dpi=300, bbox_inches="tight")
        print("  Saved: chart_2_intelligence_gap.png")
        plt.close()

    def chart_3_inference_latency(self):
        """Chart 3: Inference latency breakdown by stage"""
        print("Generating Chart 3: Inference Latency...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Data
        latencies = self.metrics["latency"]
        stages = [k for k in latencies.keys() if k != "Total (End-to-End)"]
        times = [latencies[s] for s in stages]
        times_ms = [t * 1000 for t in times]
        total_time = latencies["Total (End-to-End)"]

        # Chart 1: Horizontal bar chart
        colors_grad = [COLOR_CNN, COLOR_SCL, COLOR_ENSEMBLE, COLOR_ACCENT]
        bars = ax1.barh(
            stages,
            times_ms,
            color=colors_grad,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, times_ms)):
            ax1.text(
                val + 100,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}ms",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        ax1.set_xlabel("Time (milliseconds)", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Inference Latency per Stage\n(CPU-Only, 8GB RAM)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax1.grid(True, alpha=0.3, linestyle="--", axis="x", color=COLOR_GRID)
        ax1.set_xlim(0, max(times_ms) + 1000)

        # Chart 2: Pie chart (percentage breakdown)
        percentages = [(t / total_time) * 100 for t in times]
        explode = (0.05, 0.05, 0.05, 0.05)

        wedges, texts, autotexts = ax2.pie(
            percentages,
            labels=stages,
            autopct="%1.1f%%",
            colors=colors_grad,
            explode=explode,
            startangle=90,
            textprops={"fontsize": 10},
        )

        # Bold the percentage text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(11)

        ax2.set_title(
            "Latency Distribution\n(Total: {:.0f}ms)".format(total_time * 1000),
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        # Add summary text
        summary_text = f"Total End-to-End Latency: {total_time * 1000:.0f}ms ({total_time:.2f}s)\nSystematic Bottleneck: VLM Stage (54.3%)"
        fig.text(
            0.5,
            0.02,
            summary_text,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig("chart_3_inference_latency.png", dpi=300, bbox_inches="tight")
        print("  Saved: chart_3_inference_latency.png")
        plt.close()

    def generate_all_charts(self):
        """Generate all three charts"""
        print("\n" + "=" * 70)
        print("VISUALIZATION CODE EXECUTION: Generating Charts")
        print("=" * 70 + "\n")

        self.chart_1_accuracy_vs_rounds()
        self.chart_2_intelligence_gap()
        self.chart_3_inference_latency()

        print("\n" + "=" * 70)
        print("CHARTS GENERATION COMPLETE")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  1. chart_1_accuracy_vs_rounds.png")
        print("  2. chart_2_intelligence_gap.png")
        print("  3. chart_3_inference_latency.png")
        print("\nFiles ready for:")
        print("  - IEEE paper figures")
        print("  - Conference presentations")
        print("  - Technical documentation")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        visualizer = MetricsVisualizer()
        visualizer.generate_all_charts()
    except ImportError as e:
        print("ERROR: matplotlib not installed")
        print("To install: pip install matplotlib numpy")
        print("\nAlternatively, run: python generate_charts.py")
