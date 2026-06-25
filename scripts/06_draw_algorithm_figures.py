"""Draw candidate algorithm figures for Section 3.

The figures intentionally follow a compact ML-conference schematic style:
white background, black lines, small colored points, equations, and panel
labels. They are meant as candidates only and are not inserted into LaTeX.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "algorithm_versions"


PAL = {
    "ink": "#111111",
    "muted": "#4D4D4D",
    "blue": "#5B8FD9",
    "blue_light": "#DCEAFF",
    "orange": "#E7A53B",
    "orange_light": "#FFF0CC",
    "green": "#58A65C",
    "green_light": "#E5F4E6",
    "red": "#C44E52",
    "red_light": "#F8DDDE",
    "purple": "#8E6BBE",
    "purple_light": "#EEE6FA",
    "gray": "#F5F5F5",
    "line": "#222222",
}


def configure_style() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in names else "DejaVu Serif"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif, "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_ax(figsize=(11.5, 3.45)):
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc="white", ec=None, lw=1.2, r=0.02, fs=10, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec or PAL["line"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.22, zorder=z + 1)
    return patch


def rect(ax, x, y, w, h, text="", fc="white", ec=None, lw=1.1, fs=9, z=2):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or PAL["line"], linewidth=lw, zorder=z)
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2, zorder=z + 1)
    return patch


def arrow(ax, start, end, rad=0, lw=1.25, color=None, ms=10, z=5):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or PAL["line"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def panel_label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=11.5, fontweight="bold")


def draw_doc_stack(ax, x, y, s=1.0, label=None):
    for dx, dy in [(0, 0), (0.007 * s, 0.009 * s), (0.014 * s, 0.018 * s)]:
        ax.add_patch(
            Rectangle(
                (x + dx, y + dy),
                0.055 * s,
                0.068 * s,
                facecolor="white",
                edgecolor=PAL["line"],
                linewidth=1.1,
                zorder=2,
            )
        )
        ax.plot([x + dx + 0.012 * s, x + dx + 0.045 * s], [y + dy + 0.048 * s, y + dy + 0.048 * s], color=PAL["line"], lw=0.8)
        ax.plot([x + dx + 0.012 * s, x + dx + 0.038 * s], [y + dy + 0.034 * s, y + dy + 0.034 * s], color=PAL["line"], lw=0.8)
        ax.plot([x + dx + 0.012 * s, x + dx + 0.042 * s], [y + dy + 0.021 * s, y + dy + 0.021 * s], color=PAL["line"], lw=0.8)
    if label:
        ax.text(x + 0.038 * s, y - 0.022 * s, label, ha="center", va="top", fontsize=8.5)


def draw_points(ax, center=(0.25, 0.52), scale=1.0, seed=0, ring=True):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(18, 2))
    pts[:, 0] = pts[:, 0] * 0.045 * scale + center[0]
    pts[:, 1] = pts[:, 1] * 0.105 * scale + center[1]
    colors = [PAL["blue"]] * 9 + [PAL["orange"]] * 9
    rng.shuffle(colors)
    for (x, y), c in zip(pts, colors):
        ax.add_patch(Circle((x, y), 0.011 * scale, facecolor=c, edgecolor=PAL["line"], linewidth=0.7, zorder=5))
    q = (center[0] - 0.035 * scale, center[1] + 0.06 * scale)
    ax.add_patch(Circle(q, 0.0135 * scale, facecolor=PAL["red"], edgecolor=PAL["line"], linewidth=0.8, zorder=7))
    ax.add_patch(Circle(q, 0.006 * scale, facecolor=PAL["orange_light"], edgecolor="none", zorder=8))
    if ring:
        ax.add_patch(Ellipse((center[0] - 0.006 * scale, center[1] + 0.015 * scale), 0.15 * scale, 0.19 * scale, angle=-21, fill=False, edgecolor=PAL["line"], lw=1.1, zorder=4))
    return q


def draw_mini_table(ax, x, y, rows=4, cols=4, cw=0.015, ch=0.018):
    palette = [PAL["blue"], PAL["orange"], PAL["green"], PAL["purple"]]
    for r in range(rows):
        for c in range(cols):
            fc = "white"
            if (r + c) % 3 == 0:
                fc = palette[(r + 2 * c) % len(palette)]
            ax.add_patch(Rectangle((x + c * cw, y + r * ch), cw, ch, facecolor=fc, edgecolor=PAL["line"], linewidth=0.55, zorder=3))


def draw_encoder(ax, x, y, w=0.1, h=0.28, label="Text\nencoder"):
    rect(ax, x, y, w, h, fc="white", lw=1.1)
    xs = np.linspace(x + 0.018, x + w - 0.018, 4)
    ys = np.linspace(y + 0.055, y + h - 0.055, 3)
    for yy in ys:
        for xx in xs:
            ax.add_patch(Circle((xx, yy), 0.009, facecolor=PAL["blue_light"], edgecolor=PAL["line"], linewidth=0.6))
    for i in range(len(xs) - 1):
        for yy in ys:
            ax.plot([xs[i], xs[i + 1]], [yy, yy], color=PAL["blue"], lw=0.7, alpha=0.65)
    ax.text(x + w / 2, y + h + 0.026, label, ha="center", va="bottom", fontsize=9.2)


def save_all(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig_v1_two_panel():
    fig, ax = make_ax((11.6, 3.45))
    panel_label(ax, 0.035, 0.92, "(a) Local expression shift estimation")
    panel_label(ax, 0.535, 0.92, "(b) Response prediction and mechanism testing")
    ax.plot([0.50, 0.50], [0.09, 0.88], color="#999999", lw=0.9)

    draw_doc_stack(ax, 0.04, 0.50, 0.88, label=r"$\mathcal{D}_{pre}$")
    rect(ax, 0.047, 0.26, 0.065, 0.085, r"$x_i$", fc=PAL["orange_light"], fs=10)
    arrow(ax, (0.112, 0.57), (0.145, 0.57))
    arrow(ax, (0.112, 0.305), (0.145, 0.48), rad=-0.08)
    box(ax, 0.145, 0.47, 0.095, 0.17, "Approx.\nretrieval", fc=PAL["gray"], fs=8.7)
    arrow(ax, (0.242, 0.555), (0.286, 0.555))
    draw_points(ax, center=(0.335, 0.54), scale=0.86, seed=6)
    ax.text(0.335, 0.72, r"$k$NN context $\mathcal{N}_k(i)$", ha="center", va="bottom", fontsize=9.2)
    arrow(ax, (0.405, 0.54), (0.425, 0.54), rad=0.02)
    box(ax, 0.425, 0.455, 0.065, 0.17, r"$\Delta E_i$" + "\nmatched\nshift", fc=PAL["green_light"], fs=8.6)
    ax.text(0.275, 0.165, r"$\Delta E_i = E_i - \frac{1}{k}\sum_{j\in\mathcal{N}_k(i)} E_j$", ha="center", va="center", fontsize=11.2)

    # Panel b
    box(ax, 0.545, 0.70, 0.09, 0.10, r"$T_i$", fc="white", fs=12)
    box(ax, 0.545, 0.51, 0.09, 0.10, r"$X_i$", fc=PAL["gray"], fs=12)
    box(ax, 0.545, 0.32, 0.09, 0.10, r"$\Delta E_i,\ c_i$", fc=PAL["green_light"], fs=11)
    draw_encoder(ax, 0.685, 0.62, 0.085, 0.20, "Encoder")
    box(ax, 0.685, 0.38, 0.095, 0.12, "Feature\nfusion", fc=PAL["purple_light"], fs=8.9)
    arrow(ax, (0.635, 0.75), (0.685, 0.72))
    arrow(ax, (0.635, 0.56), (0.685, 0.44))
    arrow(ax, (0.635, 0.37), (0.685, 0.44))
    arrow(ax, (0.727, 0.62), (0.727, 0.50))
    ax.text(0.725, 0.315, r"$z_i=[h_i,X_i,\Delta E_i,c_i]$", ha="center", va="center", fontsize=11.5)
    arrow(ax, (0.78, 0.44), (0.825, 0.44))
    box(ax, 0.825, 0.335, 0.105, 0.22, "XGBoost\nLightGBM\nTabTransformer", fc="white", fs=8.7)
    arrow(ax, (0.93, 0.48), (0.968, 0.60), rad=0.1)
    arrow(ax, (0.93, 0.445), (0.968, 0.445))
    arrow(ax, (0.93, 0.41), (0.968, 0.29), rad=-0.1)
    ax.text(0.970, 0.60, r"$\hat{y}^{timely}_i$", ha="left", va="center", fontsize=11.5)
    ax.text(0.970, 0.445, r"$\hat{y}^{relief}_i$", ha="left", va="center", fontsize=11.5)
    ax.text(0.970, 0.29, r"$\hat{y}^{money}_i$", ha="left", va="center", fontsize=11.5)
    arrow(ax, (0.875, 0.335), (0.820, 0.20), rad=0.1)
    arrow(ax, (0.895, 0.335), (0.940, 0.20), rad=-0.1)
    box(ax, 0.755, 0.115, 0.105, 0.09, "SHAP\nexplanation", fc=PAL["blue_light"], fs=8.7)
    box(ax, 0.895, 0.115, 0.095, 0.09, "DML / GRF\nheterogeneity", fc=PAL["red_light"], fs=7.9)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["blue"], markeredgecolor=PAL["line"], markersize=6, label="Pre-ChatGPT"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["orange"], markeredgecolor=PAL["line"], markersize=6, label="Post-ChatGPT"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["red"], markeredgecolor=PAL["line"], markersize=6, label="Query complaint"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.185, 0.045), frameon=False, fontsize=8.2, ncol=3, handletextpad=0.35, columnspacing=0.85)
    save_all(fig, "fig2_algorithm_v1_two_panel")


def fig_v2_pipeline():
    fig, ax = make_ax((11.2, 3.6))
    panel_label(ax, 0.035, 0.92, "Compact algorithmic pipeline")

    xs = [0.05, 0.25, 0.45, 0.65, 0.84]
    y_top = 0.59
    blocks = [
        ("CFPB complaints\n$text + metadata$", PAL["gray"]),
        ("Matched context\n$\\mathcal{N}_k(i)$", "white"),
        ("Expression score\n$E_i,\\ \\Delta E_i$", PAL["green_light"]),
        ("Semantic topics\n$h_i,\\ c_i$", PAL["blue_light"]),
        ("Response model\n$\\hat{Y}_i$", "white"),
    ]
    for idx, (x, (txt, fc)) in enumerate(zip(xs, blocks)):
        box(ax, x, y_top, 0.13, 0.18, txt, fc=fc, fs=9.5)
        if idx < len(xs) - 1:
            arrow(ax, (x + 0.13, y_top + 0.09), (xs[idx + 1], y_top + 0.09))

    # lower row details
    draw_doc_stack(ax, 0.065, 0.28, 0.8, label="raw text")
    draw_points(ax, center=(0.315, 0.34), scale=0.78, seed=11)
    ax.text(0.315, 0.19, "local pre-post comparison", ha="center", fontsize=8.5)
    ax.text(0.515, 0.35, r"$E_i = f(e^{legal}_i,e^{evid}_i,e^{struct}_i,e^{temp}_i)$", ha="center", fontsize=11.5)
    draw_mini_table(ax, 0.672, 0.28, rows=4, cols=5, cw=0.014, ch=0.018)
    ax.text(0.71, 0.19, "embedding + topic id", ha="center", fontsize=8.5)
    box(ax, 0.845, 0.27, 0.055, 0.09, "SHAP", fc=PAL["purple_light"], fs=8.5)
    box(ax, 0.925, 0.27, 0.055, 0.09, "DML", fc=PAL["red_light"], fs=8.5)
    arrow(ax, (0.905, 0.36), (0.95, 0.59), rad=0.15)
    arrow(ax, (0.872, 0.36), (0.895, 0.59), rad=0.1)

    # decorative but meaningful brackets for rows
    ax.text(0.50, 0.82, "text expression estimation", ha="center", va="bottom", fontsize=10.5)
    ax.plot([0.25, 0.73], [0.81, 0.81], color=PAL["line"], lw=0.9)
    ax.text(0.86, 0.45, "prediction\nand testing", ha="center", va="center", fontsize=9)
    save_all(fig, "fig2_algorithm_v2_compact_pipeline")


def fig_v3_energy_objective():
    fig, ax = make_ax((11.4, 3.15))
    panel_label(ax, 0.035, 0.90, "Objective decomposition for expression-aware response modeling")

    rect(ax, 0.07, 0.31, 0.13, 0.36, "Complaint\nencoder", fc="white", fs=10)
    draw_mini_table(ax, 0.095, 0.39, rows=5, cols=4, cw=0.017, ch=0.022)
    ax.text(0.055, 0.49, r"$T_i, X_i$", ha="right", va="center", fontsize=12)
    arrow(ax, (0.057, 0.49), (0.07, 0.49))
    ax.text(0.205, 0.49, r"$z_i$", ha="left", va="center", fontsize=12)

    arrow(ax, (0.20, 0.53), (0.30, 0.72), rad=0.2)
    arrow(ax, (0.20, 0.45), (0.30, 0.26), rad=-0.2)
    box(ax, 0.31, 0.66, 0.13, 0.13, "Expression\nhead", fc=PAL["green_light"], fs=10)
    box(ax, 0.31, 0.20, 0.13, 0.13, "Response\nhead", fc=PAL["blue_light"], fs=10)
    arrow(ax, (0.44, 0.725), (0.52, 0.725))
    arrow(ax, (0.44, 0.265), (0.52, 0.265))
    ax.text(0.535, 0.725, r"$E(T_i)$", ha="left", va="center", fontsize=14)
    ax.text(0.535, 0.265, r"$\ell(Y_i\mid T_i,X_i)$", ha="left", va="center", fontsize=14)
    ax.text(0.535, 0.64, "expression energy", ha="left", fontsize=9.5)
    ax.text(0.535, 0.18, "response loss", ha="left", fontsize=9.5)

    arrow(ax, (0.66, 0.70), (0.79, 0.58), rad=-0.22)
    arrow(ax, (0.66, 0.28), (0.79, 0.42), rad=0.22)
    box(ax, 0.79, 0.39, 0.14, 0.18, r"$\mathcal{L}$" + "\nexpression-aware\nobjective", fc="white", fs=10)
    ax.text(0.855, 0.27, r"$\mathcal{L}= \ell(Y_i\mid z_i)+\lambda\Vert\Delta E_i\Vert$", ha="center", va="center", fontsize=13)
    arrow(ax, (0.93, 0.48), (0.985, 0.48))
    ax.text(0.988, 0.48, r"$\hat{Y}_i,\ \tau(X_i)$", ha="left", va="center", fontsize=13)

    # Small local contrast inset
    box(ax, 0.07, 0.08, 0.26, 0.12, r"local contrast:  $\Delta E_i = E_i - \mathrm{mean}_{j\in\mathcal{N}_k(i)}E_j$", fc=PAL["gray"], fs=9)
    arrow(ax, (0.25, 0.20), (0.33, 0.66), rad=-0.25, lw=1.0)
    save_all(fig, "fig2_algorithm_v3_objective")


def main() -> None:
    configure_style()
    fig_v1_two_panel()
    fig_v2_pipeline()
    fig_v3_energy_objective()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
