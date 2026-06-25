"""Draw Figure 1: conceptual mechanism and analysis framework.

This is a publication-style schematic, not a statistical chart. It follows the
same visual QA principles used for data figures: fixed final size, colorblind-
safe colors, vector export, CJK font configuration, and a PNG preview.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PROFILE_PATH = ROOT / "outputs" / "tables" / "dataset_profile.json"


COLORS = {
    "ink": "#1E2A32",
    "muted": "#5C6670",
    "blue": "#4C78A8",
    "blue_light": "#DCEAF7",
    "teal": "#72B7B2",
    "teal_light": "#E1F3F1",
    "orange": "#F58518",
    "orange_light": "#FFF0DA",
    "green": "#54A24B",
    "green_light": "#E6F2E4",
    "purple": "#B279A2",
    "purple_light": "#F2E7F0",
    "red": "#E45756",
    "red_light": "#FBE3E2",
    "gray": "#D9DEE3",
    "gray_light": "#F7F9FA",
    "border": "#6D7D8C",
}


def configure_style() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["Microsoft YaHei", "SimHei", "SimSun"]:
        if font in names:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    text=None,
    fc="#FFFFFF",
    ec=None,
    lw=1.2,
    radius=0.02,
    ls="-",
    alpha=1.0,
    z=2,
    fontsize=8,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or COLORS["border"],
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=COLORS["ink"],
            linespacing=1.35,
            zorder=z + 1,
        )
    return patch


def arrow(ax, start, end, color=None, lw=1.4, style="-|>", rad=0.0, z=5):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        color=color or COLORS["border"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def panel_label(ax, x, y, label):
    ax.text(
        x,
        y,
        label,
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
        zorder=10,
    )


def small_doc(ax, x, y, w, h, fc="#FFFFFF", ec=None, lines=4):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec or COLORS["blue"], lw=1.0, radius=0.008)
    for i in range(lines):
        yy = y + h * (0.72 - i * 0.15)
        ax.add_line(
            Line2D(
                [x + 0.12 * w, x + 0.82 * w],
                [yy, yy],
                lw=1.0,
                color=COLORS["blue"],
                alpha=0.75,
                zorder=4,
            )
        )


def draw_matrix(ax, x, y, w, h, seed=2):
    rng = np.random.default_rng(seed)
    rows, cols = 7, 9
    cell_w, cell_h = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            value = rng.random()
            if value > 0.76:
                color = COLORS["teal"] if value < 0.90 else COLORS["orange"]
            else:
                color = "#FFFFFF"
            ax.add_patch(
                Rectangle(
                    (x + c * cell_w, y + (rows - 1 - r) * cell_h),
                    cell_w * 0.94,
                    cell_h * 0.88,
                    facecolor=color,
                    edgecolor="#B8C3CC",
                    linewidth=0.35,
                    zorder=3,
                )
            )


def draw_embedding(ax, x, y, w, h, seed=4):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.25, 0.70], [0.63, 0.70], [0.34, 0.30], [0.72, 0.31]])
    colors = [COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["purple"]]
    for center, color in zip(centers, colors):
        pts = rng.normal(center, [0.055, 0.055], size=(18, 2))
        ax.scatter(
            x + pts[:, 0] * w,
            y + pts[:, 1] * h,
            s=10,
            color=color,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )


def draw_network(ax, x, y, w, h):
    nodes = np.array(
        [
            [0.20, 0.70],
            [0.50, 0.82],
            [0.78, 0.63],
            [0.30, 0.35],
            [0.65, 0.28],
        ]
    )
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (1, 3), (0, 4)]
    for a, b in edges:
        ax.add_line(
            Line2D(
                [x + nodes[a, 0] * w, x + nodes[b, 0] * w],
                [y + nodes[a, 1] * h, y + nodes[b, 1] * h],
                color=COLORS["green"],
                linewidth=1.0,
                alpha=0.70,
                zorder=3,
            )
        )
    for i, node in enumerate(nodes):
        ax.add_patch(
            Circle(
                (x + node[0] * w, y + node[1] * h),
                radius=w * 0.055,
                facecolor=[COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["purple"], COLORS["green"]][i],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        )


def draw_model(ax, x, y, w, h):
    xs = [x + 0.15 * w, x + 0.45 * w, x + 0.75 * w]
    layer_counts = [4, 5, 3]
    for li, (xx, count) in enumerate(zip(xs, layer_counts)):
        ys = np.linspace(y + 0.20 * h, y + 0.82 * h, count)
        for yy in ys:
            ax.add_patch(
                Circle(
                    (xx, yy),
                    radius=0.013,
                    facecolor=COLORS["blue_light"] if li != 1 else COLORS["purple_light"],
                    edgecolor=COLORS["blue"] if li != 1 else COLORS["purple"],
                    linewidth=0.8,
                    zorder=4,
                )
            )
    for i in range(len(xs) - 1):
        ys1 = np.linspace(y + 0.20 * h, y + 0.82 * h, layer_counts[i])
        ys2 = np.linspace(y + 0.20 * h, y + 0.82 * h, layer_counts[i + 1])
        for yy1 in ys1:
            for yy2 in ys2:
                ax.add_line(
                    Line2D(
                        [xs[i], xs[i + 1]],
                        [yy1, yy2],
                        color="#AEB8C2",
                        linewidth=0.35,
                        alpha=0.45,
                        zorder=2,
                    )
                )


def draw_bars(ax, x, y, w, h):
    labels = ["法律术语", "证据词", "模板短语", "主题风险"]
    values = [0.92, 0.76, 0.66, 0.55]
    colors = [COLORS["purple"], COLORS["teal"], COLORS["orange"], COLORS["blue"]]
    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        yy = y + h * (0.77 - i * 0.20)
        ax.add_patch(
            Rectangle(
                (x + 0.40 * w, yy - 0.018),
                w * 0.48 * val,
                0.030,
                facecolor=color,
                edgecolor="none",
                alpha=0.82,
                zorder=4,
            )
        )
        ax.text(x + 0.03 * w, yy, lab, fontsize=6.4, ha="left", va="center", color=COLORS["ink"])
    ax.text(x + 0.03 * w, y + 0.93 * h, "可解释信号", fontsize=7.4, fontweight="bold", ha="left")


def load_profile() -> dict:
    if PROFILE_PATH.exists():
        with PROFILE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    profile = load_profile()

    # Wide and moderately shallow: suitable for a full-width introductory figure.
    fig = plt.figure(figsize=(7.6, 3.85), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(
        ax,
        0.025,
        0.035,
        0.950,
        0.915,
        fc="#FFFFFF",
        ec="#B78A9C",
        lw=1.6,
        radius=0.03,
        ls=(0, (1.4, 2.0)),
        z=1,
    )

    ax.text(
        0.050,
        0.925,
        "LLM 影响金融投诉表达与企业响应的概念框架",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["ink"],
    )
    ax.text(
        0.950,
        0.925,
        "CFPB consumer complaints",
        fontsize=7.5,
        ha="right",
        va="center",
        color=COLORS["muted"],
    )

    # Panel a: data and setting.
    panel_label(ax, 0.055, 0.875, "a")
    rounded_box(ax, 0.075, 0.665, 0.220, 0.195, fc=COLORS["blue_light"], ec=COLORS["blue"], lw=1.2)
    ax.text(0.185, 0.830, "公开投诉数据", fontsize=9, fontweight="bold", ha="center", color=COLORS["ink"])
    small_doc(ax, 0.095, 0.700, 0.045, 0.090, fc="#FFFFFF", ec=COLORS["blue"])
    small_doc(ax, 0.118, 0.715, 0.045, 0.090, fc="#FFFFFF", ec=COLORS["teal"])
    small_doc(ax, 0.141, 0.730, 0.045, 0.090, fc="#FFFFFF", ec=COLORS["orange"])
    date_min = str(profile.get("date_min", "2015"))[:7].replace("-", ".")
    date_max = str(profile.get("date_max", "2026"))[:7].replace("-", ".")
    data_text = (
        f"{profile.get('rows', 30263):,} 条叙事投诉\n"
        f"{date_min}--{date_max}\n"
        f"{profile.get('unique_products', 17)} 类产品 / {profile.get('unique_issues', 63)} 类问题"
    )
    ax.text(0.243, 0.750, data_text, fontsize=6.5, ha="center", va="center", linespacing=1.35, color=COLORS["ink"])

    # Panel b: LLM shock and expression mechanism.
    panel_label(ax, 0.330, 0.875, "b")
    rounded_box(ax, 0.350, 0.665, 0.285, 0.195, fc=COLORS["purple_light"], ec=COLORS["purple"], lw=1.2)
    ax.text(0.492, 0.830, "LLM 可用性提升", fontsize=9, fontweight="bold", ha="center", color=COLORS["ink"])
    ax.text(0.392, 0.805, "ChatGPT\n发布后", fontsize=6.7, ha="center", va="center", color=COLORS["purple"])
    rounded_box(ax, 0.455, 0.770, 0.125, 0.043, "写作辅助", fc="#FFFFFF", ec=COLORS["purple"], lw=1.0, radius=0.012)
    arrow(ax, (0.418, 0.775), (0.452, 0.788), color=COLORS["purple"], lw=1.2)
    for i, (txt, col) in enumerate(
        [
            ("结构化叙事", COLORS["blue"]),
            ("法律化表达", COLORS["purple"]),
            ("证据组织", COLORS["teal"]),
            ("模板化短语", COLORS["orange"]),
        ]
    ):
        x0 = 0.370 + (i % 2) * 0.130
        y0 = 0.680 + (1 - i // 2) * 0.050
        rounded_box(ax, x0, y0, 0.115, 0.036, txt, fc="#FFFFFF", ec=col, lw=0.9, radius=0.010)

    # Panel c: algorithmic representation.
    panel_label(ax, 0.665, 0.875, "c")
    rounded_box(ax, 0.685, 0.665, 0.235, 0.195, fc=COLORS["teal_light"], ec=COLORS["teal"], lw=1.2)
    ax.text(0.803, 0.830, "文本表示与主题发现", fontsize=9, fontweight="bold", ha="center", color=COLORS["ink"])
    draw_matrix(ax, 0.705, 0.705, 0.075, 0.080)
    draw_embedding(ax, 0.810, 0.700, 0.090, 0.090)
    arrow(ax, (0.786, 0.745), (0.807, 0.745), color=COLORS["teal"], lw=1.1)
    ax.text(0.742, 0.685, "特征矩阵", fontsize=6.5, ha="center", color=COLORS["muted"])
    ax.text(0.855, 0.685, "风险主题", fontsize=6.5, ha="center", color=COLORS["muted"])

    # Cross-panel arrows in the top row.
    arrow(ax, (0.300, 0.762), (0.345, 0.762), color=COLORS["border"], lw=1.5)
    arrow(ax, (0.640, 0.762), (0.680, 0.762), color=COLORS["border"], lw=1.5)

    # Panel d: firm response mechanism.
    panel_label(ax, 0.055, 0.610, "d")
    rounded_box(ax, 0.075, 0.360, 0.460, 0.230, fc=COLORS["gray_light"], ec=COLORS["border"], lw=1.2)
    ax.text(0.305, 0.560, "企业理解、分类与回应机制", fontsize=9.2, fontweight="bold", ha="center", color=COLORS["ink"])
    rounded_box(ax, 0.105, 0.435, 0.105, 0.085, "投诉文本\n与证据线索", fc="#FFFFFF", ec=COLORS["blue"], lw=1.0, radius=0.012)
    rounded_box(ax, 0.245, 0.435, 0.105, 0.085, "内部审核\n/自动分派", fc="#FFFFFF", ec=COLORS["teal"], lw=1.0, radius=0.012)
    rounded_box(ax, 0.385, 0.435, 0.105, 0.085, "响应策略\n与救济判断", fc="#FFFFFF", ec=COLORS["orange"], lw=1.0, radius=0.012)
    arrow(ax, (0.213, 0.477), (0.242, 0.477), color=COLORS["border"], lw=1.2)
    arrow(ax, (0.353, 0.477), (0.382, 0.477), color=COLORS["border"], lw=1.2)
    rounded_box(ax, 0.110, 0.375, 0.380, 0.035, "产品类别 / 投诉主题 / 公司差异 / 地区差异 作为调节因素", fc="#FFFFFF", ec=COLORS["gray"], lw=0.8, radius=0.010)

    # Panel e: outcome and explainable model.
    panel_label(ax, 0.575, 0.610, "e")
    rounded_box(ax, 0.595, 0.360, 0.325, 0.230, fc=COLORS["green_light"], ec=COLORS["green"], lw=1.2)
    ax.text(0.758, 0.560, "可解释预测与结果变量", fontsize=9.2, fontweight="bold", ha="center", color=COLORS["ink"])
    draw_model(ax, 0.615, 0.420, 0.115, 0.095)
    draw_bars(ax, 0.735, 0.405, 0.120, 0.115)
    ax.text(0.672, 0.385, "TF-IDF + 文本特征\n主题 + 结构变量", fontsize=6.2, ha="center", va="top", color=COLORS["muted"])
    ax.text(0.885, 0.505, "响应结果", fontsize=6.6, fontweight="bold", ha="center", color=COLORS["ink"])
    for i, (txt, col) in enumerate(
        [
            ("金钱救济", COLORS["orange"]),
            ("非金钱救济", COLORS["green"]),
            ("及时回应", COLORS["blue"]),
        ]
    ):
        rounded_box(
            ax,
            0.855,
            0.458 - i * 0.034,
            0.060,
            0.026,
            txt,
            fc="#FFFFFF",
            ec=col,
            lw=0.75,
            radius=0.007,
            fontsize=5.6,
        )
    arrow(ax, (0.540, 0.475), (0.590, 0.475), color=COLORS["border"], lw=1.5)

    # Bottom mini panels: empirical checks.
    rounded_box(ax, 0.075, 0.105, 0.845, 0.205, fc="#FFFFFF", ec="#B78A9C", lw=1.0, radius=0.025, ls=(0, (1.2, 2.0)))
    ax.text(0.100, 0.275, "经验检验路径", fontsize=9.2, fontweight="bold", ha="left", color=COLORS["ink"])
    checks = [
        ("时间比较", "ChatGPT 前后\n表达特征变化", COLORS["purple"], COLORS["purple_light"]),
        ("主题异质性", "不同风险主题\n响应差异", COLORS["teal"], COLORS["teal_light"]),
        ("产品异质性", "征信 / 银行账户\n信用卡等", COLORS["blue"], COLORS["blue_light"]),
        ("模型解释", "识别与救济相关的\n文本信号", COLORS["orange"], COLORS["orange_light"]),
    ]
    for i, (head, body, ec, fc) in enumerate(checks):
        x0 = 0.115 + i * 0.198
        rounded_box(ax, x0, 0.145, 0.155, 0.095, fc=fc, ec=ec, lw=1.0, radius=0.016)
        ax.text(x0 + 0.0775, 0.215, head, fontsize=7.5, fontweight="bold", ha="center", color=COLORS["ink"])
        ax.text(x0 + 0.0775, 0.175, body, fontsize=6.8, ha="center", va="center", color=COLORS["ink"], linespacing=1.25)
        if i < len(checks) - 1:
            arrow(ax, (x0 + 0.160, 0.192), (x0 + 0.193, 0.192), color=COLORS["border"], lw=1.0)

    # A thin legend-style annotation.
    ax.text(
        0.500,
        0.065,
        "注：图中箭头表示研究假设中的信息流和分析流，不直接表示因果效应。",
        fontsize=6.7,
        color=COLORS["muted"],
        ha="center",
        va="center",
    )

    pdf_path = FIG_DIR / "fig1_mechanism_framework.pdf"
    svg_path = FIG_DIR / "fig1_mechanism_framework.svg"
    png_path = FIG_DIR / "fig1_mechanism_framework.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04, dpi=350)
    print(pdf_path)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
