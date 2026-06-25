"""Draw compact mechanism-figure alternatives for the CFPB-LLM paper.

The goal is a paper-style framework figure: compact, meaningful vector stickers,
pastel gradients, tight module connections, and readable Chinese labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Circle,
    Rectangle,
    Polygon,
)
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "mechanism_versions"
PROFILE_PATH = ROOT / "outputs" / "tables" / "dataset_profile.json"


PAL = {
    "ink": "#23313B",
    "muted": "#68747F",
    "blue": "#4E79A7",
    "blue2": "#A9C8E8",
    "blue_bg": "#EAF3FB",
    "teal": "#59AFA8",
    "teal2": "#B8E0DD",
    "teal_bg": "#E9F7F5",
    "green": "#6DAA57",
    "green2": "#CDE8C7",
    "green_bg": "#F0F8ED",
    "orange": "#F28E2B",
    "orange2": "#F8C999",
    "orange_bg": "#FFF2E4",
    "purple": "#B07AA1",
    "purple2": "#DFC2D6",
    "purple_bg": "#F6ECF3",
    "yellow": "#E6B93E",
    "yellow_bg": "#FFF7D9",
    "red": "#D65F5F",
    "red_bg": "#FBEAEA",
    "line": "#5E6B73",
    "grid": "#B8C2C9",
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def c(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def left(self):
        return (self.x, self.y + self.h / 2)

    @property
    def right(self):
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def top(self):
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def bottom(self):
        return (self.x + self.w / 2, self.y)


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


def load_profile() -> dict:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {}


def make_ax(figsize=(9.2, 5.25)):
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def gradient_rect(ax, box: Box, c1: str, c2: str, radius=0.018, ec=None, lw=1.2, z=1):
    """Pastel horizontal gradient clipped to a rounded box."""
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor="none",
        edgecolor=ec or PAL["line"],
        linewidth=lw,
        zorder=z + 2,
    )
    ax.add_patch(patch)
    import matplotlib.colors as mcolors

    n = 256
    rgb1 = np.array(mcolors.to_rgb(c1))
    rgb2 = np.array(mcolors.to_rgb(c2))
    grad = np.linspace(0, 1, n)
    arr = (rgb1[None, :] * (1 - grad[:, None]) + rgb2[None, :] * grad[:, None])
    arr = np.repeat(arr[None, :, :], 3, axis=0)
    im = ax.imshow(
        arr,
        extent=(box.x, box.x + box.w, box.y, box.y + box.h),
        origin="lower",
        aspect="auto",
        zorder=z,
    )
    im.set_clip_path(patch)
    return patch


def label_bar(ax, box: Box, text: str, color: str, fontsize=9.5):
    bar = Box(box.x + box.w * 0.22, box.y + box.h - 0.038, box.w * 0.56, 0.046)
    gradient_rect(ax, bar, "#FFFFFF", color, radius=0.010, ec=color, lw=0.8, z=10)
    ax.text(bar.x + bar.w / 2, bar.y + bar.h / 2, text, fontsize=fontsize, fontweight="bold", ha="center", va="center", color=PAL["ink"], zorder=20)


def small_box(ax, box: Box, text: str, fc="#FFFFFF", ec=None, fontsize=7.0, lw=0.8, z=20):
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=fc,
        edgecolor=ec or PAL["line"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        box.x + box.w / 2,
        box.y + box.h / 2,
        text,
        fontsize=fontsize,
        ha="center",
        va="center",
        color=PAL["ink"],
        linespacing=1.15,
        zorder=z + 1,
    )
    return patch


def text(ax, x, y, s, size=7, weight="normal", ha="center", va="center", color=None):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color or PAL["ink"], linespacing=1.15, zorder=50)


def arrow(ax, p1, p2, color=None, lw=1.6, rad=0.0, z=30, ms=12, alpha=1.0):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or PAL["line"],
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def thick_arrow(ax, p1, p2, color):
    arrow(ax, (p1[0] + 0.003, p1[1] - 0.003), (p2[0] + 0.003, p2[1] - 0.003), color="#D8DEE4", lw=3.0, ms=16, z=15)
    arrow(ax, p1, p2, color=color, lw=2.2, ms=16, z=35)


def panel(ax, box: Box, title: str, c1: str, c2: str, ec: str, tag: str | None = None):
    gradient_rect(ax, box, c1, c2, radius=0.016, ec=ec, lw=1.15, z=2)
    label_bar(ax, box, title, ec)
    if tag:
        ax.text(box.x + 0.010, box.y + box.h - 0.032, tag, fontsize=9.5, fontweight="bold", ha="left", va="top", color=PAL["ink"], zorder=30)


def draw_database_icon(ax, x, y, s=1.0, color=PAL["blue"]):
    w, h = 0.048 * s, 0.062 * s
    for i in range(3):
        yy = y + i * h * 0.23
        ax.add_patch(Rectangle((x, yy), w, h * 0.28, facecolor="#FFFFFF", edgecolor=color, linewidth=0.8, zorder=25))
        ax.add_patch(Circle((x + w / 2, yy + h * 0.28), w / 2, facecolor="#FFFFFF", edgecolor=color, linewidth=0.8, zorder=26))
    text(ax, x + w / 2, y - 0.006, "DB", size=5.5, color=color)


def draw_documents_icon(ax, x, y, s=1.0, color=PAL["blue"]):
    w, h = 0.052 * s, 0.066 * s
    for i, ec in enumerate([PAL["teal"], PAL["orange"], color]):
        dx, dy = i * 0.008 * s, i * 0.008 * s
        small_box(ax, Box(x + dx, y + dy, w, h), "", fc="#FFFFFF", ec=ec, lw=0.8, z=20)
        for j in range(4):
            ax.add_line(Line2D([x + dx + w * 0.18, x + dx + w * 0.80], [y + dy + h * (0.72 - j * 0.15)] * 2, lw=0.9, color=ec, alpha=0.75, zorder=25))


def draw_chat_pen_icon(ax, x, y, s=1.0):
    bubble = FancyBboxPatch((x, y + 0.018 * s), 0.070 * s, 0.050 * s, boxstyle="round,pad=0.004,rounding_size=0.013", facecolor="#FFFFFF", edgecolor=PAL["purple"], linewidth=0.9, zorder=22)
    ax.add_patch(bubble)
    tri = Polygon([[x + 0.014 * s, y + 0.020 * s], [x + 0.026 * s, y + 0.010 * s], [x + 0.030 * s, y + 0.021 * s]], closed=True, facecolor="#FFFFFF", edgecolor=PAL["purple"], linewidth=0.8, zorder=21)
    ax.add_patch(tri)
    text(ax, x + 0.035 * s, y + 0.044 * s, "LLM", size=6.2, weight="bold", color=PAL["purple"])
    ax.add_patch(Rectangle((x + 0.046 * s, y), 0.043 * s, 0.009 * s, angle=22, facecolor=PAL["orange2"], edgecolor=PAL["orange"], linewidth=0.7, zorder=26))
    ax.add_patch(Polygon([[x + 0.087 * s, y + 0.018 * s], [x + 0.101 * s, y + 0.022 * s], [x + 0.091 * s, y + 0.008 * s]], facecolor=PAL["orange"], edgecolor=PAL["orange"], zorder=27))


def draw_scale_icon(ax, x, y, s=1.0):
    ax.add_line(Line2D([x, x + 0.070 * s], [y + 0.060 * s, y + 0.060 * s], color=PAL["purple"], lw=1.2, zorder=24))
    ax.add_line(Line2D([x + 0.035 * s, x + 0.035 * s], [y + 0.015 * s, y + 0.075 * s], color=PAL["purple"], lw=1.2, zorder=24))
    ax.add_patch(Polygon([[x + 0.018 * s, y + 0.050 * s], [x + 0.006 * s, y + 0.025 * s], [x + 0.030 * s, y + 0.025 * s]], facecolor="#FFFFFF", edgecolor=PAL["purple"], linewidth=0.8, zorder=25))
    ax.add_patch(Polygon([[x + 0.052 * s, y + 0.050 * s], [x + 0.040 * s, y + 0.025 * s], [x + 0.064 * s, y + 0.025 * s]], facecolor="#FFFFFF", edgecolor=PAL["purple"], linewidth=0.8, zorder=25))
    ax.add_patch(Rectangle((x + 0.022 * s, y + 0.008 * s), 0.026 * s, 0.007 * s, facecolor=PAL["purple2"], edgecolor=PAL["purple"], linewidth=0.5, zorder=25))


def draw_folder_icon(ax, x, y, s=1.0):
    ax.add_patch(Polygon([[x, y], [x + 0.087 * s, y], [x + 0.087 * s, y + 0.052 * s], [x + 0.034 * s, y + 0.052 * s], [x + 0.025 * s, y + 0.064 * s], [x, y + 0.064 * s]], closed=True, facecolor=PAL["yellow_bg"], edgecolor=PAL["yellow"], linewidth=0.9, zorder=20))
    for i in range(3):
        ax.add_line(Line2D([x + 0.017 * s, x + 0.070 * s], [y + 0.040 * s - i * 0.012 * s] * 2, lw=0.9, color=PAL["yellow"], zorder=24))


def draw_matrix_icon(ax, x, y, s=1.0):
    rows, cols = 5, 7
    rng = np.random.default_rng(3)
    w, h = 0.085 * s, 0.060 * s
    for r in range(rows):
        for c in range(cols):
            val = rng.random()
            col = "#FFFFFF"
            if val > 0.82:
                col = PAL["orange2"]
            elif val > 0.64:
                col = PAL["teal2"]
            ax.add_patch(Rectangle((x + c * w / cols, y + r * h / rows), w / cols * 0.92, h / rows * 0.88, facecolor=col, edgecolor=PAL["grid"], linewidth=0.35, zorder=24))


def draw_topic_clusters(ax, x, y, s=1.0):
    rng = np.random.default_rng(5)
    centers = [(0.20, 0.70, PAL["blue"]), (0.65, 0.72, PAL["teal"]), (0.32, 0.28, PAL["orange"]), (0.78, 0.32, PAL["purple"])]
    for cx, cy, col in centers:
        pts = rng.normal([cx, cy], [0.045, 0.045], (11, 2))
        ax.scatter(x + pts[:, 0] * 0.105 * s, y + pts[:, 1] * 0.075 * s, s=10 * s, color=col, alpha=0.88, edgecolor="white", linewidth=0.25, zorder=25)


def draw_network_icon(ax, x, y, s=1.0):
    layers = [(0, 4, PAL["blue"]), (0.045, 5, PAL["purple"]), (0.090, 3, PAL["teal"])]
    coords = []
    for dx, n, col in layers:
        ys = np.linspace(y + 0.010 * s, y + 0.070 * s, n)
        layer = []
        for yy in ys:
            cx = x + dx * s
            ax.add_patch(Circle((cx, yy), 0.0075 * s, facecolor="#FFFFFF", edgecolor=col, linewidth=0.9, zorder=27))
            layer.append((cx, yy))
        coords.append(layer)
    for l1, l2 in zip(coords, coords[1:]):
        for p1 in l1:
            for p2 in l2:
                ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], lw=0.25, color=PAL["grid"], alpha=0.70, zorder=23))


def draw_company_icon(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x, y), 0.075 * s, 0.075 * s, facecolor="#FFFFFF", edgecolor=PAL["green"], linewidth=0.9, zorder=23))
    ax.add_patch(Polygon([[x - 0.006 * s, y + 0.075 * s], [x + 0.0375 * s, y + 0.103 * s], [x + 0.081 * s, y + 0.075 * s]], facecolor=PAL["green2"], edgecolor=PAL["green"], linewidth=0.8, zorder=24))
    for i in range(3):
        for j in range(2):
            ax.add_patch(Rectangle((x + 0.013 * s + i * 0.019 * s, y + 0.018 * s + j * 0.023 * s), 0.010 * s, 0.012 * s, facecolor=PAL["green_bg"], edgecolor=PAL["green"], linewidth=0.45, zorder=25))


def draw_outcome_icons(ax, x, y, s=1.0):
    # coins
    for i in range(3):
        ax.add_patch(Circle((x + 0.016 * i * s, y + 0.042 * s), 0.012 * s, facecolor=PAL["orange2"], edgecolor=PAL["orange"], linewidth=0.7, zorder=24))
    # check
    ax.add_patch(Circle((x + 0.067 * s, y + 0.042 * s), 0.014 * s, facecolor=PAL["green2"], edgecolor=PAL["green"], linewidth=0.7, zorder=24))
    ax.add_line(Line2D([x + 0.060 * s, x + 0.066 * s, x + 0.076 * s], [y + 0.041 * s, y + 0.034 * s, y + 0.049 * s], color=PAL["green"], lw=1.0, zorder=26))
    # clock
    ax.add_patch(Circle((x + 0.111 * s, y + 0.042 * s), 0.014 * s, facecolor=PAL["blue_bg"], edgecolor=PAL["blue"], linewidth=0.7, zorder=24))
    ax.add_line(Line2D([x + 0.111 * s, x + 0.111 * s], [y + 0.042 * s, y + 0.053 * s], color=PAL["blue"], lw=0.8, zorder=25))
    ax.add_line(Line2D([x + 0.111 * s, x + 0.119 * s], [y + 0.042 * s, y + 0.039 * s], color=PAL["blue"], lw=0.8, zorder=25))


def draw_barmini(ax, x, y, s=1.0):
    labs = ["法律", "证据", "模板", "主题"]
    vals = [0.92, 0.76, 0.64, 0.56]
    cols = [PAL["purple"], PAL["teal"], PAL["orange"], PAL["blue"]]
    for i, (lab, val, col) in enumerate(zip(labs, vals, cols)):
        yy = y + (3 - i) * 0.017 * s
        ax.add_patch(Rectangle((x + 0.040 * s, yy), 0.075 * val * s, 0.010 * s, facecolor=col, edgecolor="none", alpha=0.78, zorder=24))
        text(ax, x, yy + 0.005 * s, lab, size=5.1 * s, ha="left")


def add_footer(ax, note="箭头表示分析流与机制假设，不直接表示因果效应。"):
    text(ax, 0.50, 0.035, note, size=6.3, color=PAL["muted"])


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("pdf", None), ("svg", None), ("png", 360)]:
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
        print(path)
    plt.close(fig)


def version_a(profile):
    """2x2 compact framework like the user's reference image."""
    fig, ax = make_ax()
    outer = Box(0.035, 0.055, 0.930, 0.885)
    gradient_rect(ax, outer, "#FFFFFF", "#FAFCFF", radius=0.018, ec="#AAB6C1", lw=1.1, z=0)
    text(ax, 0.50, 0.948, "LLM 时代金融投诉表达变化与企业响应机制框架", size=13, weight="bold")

    b1 = Box(0.060, 0.595, 0.410, 0.275)
    b2 = Box(0.530, 0.595, 0.410, 0.275)
    b3 = Box(0.060, 0.170, 0.410, 0.355)
    b4 = Box(0.530, 0.170, 0.410, 0.355)

    panel(ax, b1, "数据基础：CFPB 投诉场景", PAL["blue_bg"], "#F7FBFF", PAL["blue"], "a")
    panel(ax, b2, "表达机制：LLM 辅助投诉写作", PAL["purple_bg"], "#FFF7FC", PAL["purple"], "b")
    panel(ax, b3, "算法表征：特征、主题与预测", PAL["teal_bg"], "#F8FFFE", PAL["teal"], "c")
    panel(ax, b4, "响应分析：企业救济与异质性", PAL["orange_bg"], "#FFFDF7", PAL["orange"], "d")

    # a internals
    draw_documents_icon(ax, b1.x + 0.028, b1.y + 0.095, 1.25)
    draw_database_icon(ax, b1.x + 0.130, b1.y + 0.108, 1.1)
    date_min = str(profile.get("date_min", "2015"))[:7].replace("-", ".")
    date_max = str(profile.get("date_max", "2026"))[:7].replace("-", ".")
    small_box(ax, Box(b1.x + 0.220, b1.y + 0.150, 0.155, 0.055), f"{profile.get('rows', 30263):,} 条叙事投诉", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.4)
    small_box(ax, Box(b1.x + 0.220, b1.y + 0.085, 0.155, 0.055), f"{date_min}--{date_max}", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.4)
    small_box(ax, Box(b1.x + 0.220, b1.y + 0.020, 0.155, 0.055), f"{profile.get('unique_products', 17)} 类产品 / {profile.get('unique_issues', 63)} 类问题", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.0)
    text(ax, b1.x + 0.077, b1.y + 0.045, "投诉叙事\n企业回应\n产品/地区", size=6.2)

    # b internals
    draw_chat_pen_icon(ax, b2.x + 0.034, b2.y + 0.128, 1.30)
    draw_scale_icon(ax, b2.x + 0.160, b2.y + 0.128, 1.18)
    draw_folder_icon(ax, b2.x + 0.285, b2.y + 0.126, 1.05)
    small_box(ax, Box(b2.x + 0.030, b2.y + 0.045, 0.092, 0.048), "结构化\n叙事", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.4)
    small_box(ax, Box(b2.x + 0.135, b2.y + 0.045, 0.092, 0.048), "法律化\n表达", fc="#FFFFFF", ec=PAL["purple"], fontsize=6.4)
    small_box(ax, Box(b2.x + 0.240, b2.y + 0.045, 0.092, 0.048), "证据\n组织", fc="#FFFFFF", ec=PAL["yellow"], fontsize=6.4)
    small_box(ax, Box(b2.x + 0.335, b2.y + 0.045, 0.058, 0.048), "模板\n短语", fc="#FFFFFF", ec=PAL["orange"], fontsize=6.0)

    # c internals
    draw_matrix_icon(ax, b3.x + 0.030, b3.y + 0.210, 1.28)
    draw_topic_clusters(ax, b3.x + 0.170, b3.y + 0.205, 1.25)
    draw_network_icon(ax, b3.x + 0.305, b3.y + 0.203, 1.05)
    thick_arrow(ax, (b3.x + 0.130, b3.y + 0.240), (b3.x + 0.164, b3.y + 0.240), PAL["teal"])
    thick_arrow(ax, (b3.x + 0.270, b3.y + 0.240), (b3.x + 0.300, b3.y + 0.240), PAL["teal"])
    for i, lab in enumerate(["文本统计", "主题建模", "可解释预测"]):
        small_box(ax, Box(b3.x + 0.032 + i * 0.127, b3.y + 0.105, 0.104, 0.055), lab, fc="#FFFFFF", ec=[PAL["blue"], PAL["teal"], PAL["purple"]][i], fontsize=6.4)
    small_box(ax, Box(b3.x + 0.057, b3.y + 0.030, 0.307, 0.048), "输入：TF-IDF、法律词、证据词、主题权重、结构变量", fc="#FFFFFF", ec=PAL["grid"], fontsize=6.1)

    # d internals
    draw_company_icon(ax, b4.x + 0.035, b4.y + 0.235, 1.15)
    draw_outcome_icons(ax, b4.x + 0.160, b4.y + 0.255, 1.18)
    draw_barmini(ax, b4.x + 0.296, b4.y + 0.235, 1.25)
    small_box(ax, Box(b4.x + 0.030, b4.y + 0.147, 0.108, 0.050), "企业理解\n与分派", fc="#FFFFFF", ec=PAL["green"], fontsize=6.3)
    small_box(ax, Box(b4.x + 0.152, b4.y + 0.147, 0.108, 0.050), "救济结果\n/及时回应", fc="#FFFFFF", ec=PAL["orange"], fontsize=6.3)
    small_box(ax, Box(b4.x + 0.274, b4.y + 0.147, 0.108, 0.050), "特征解释\n/机制检验", fc="#FFFFFF", ec=PAL["purple"], fontsize=6.3)
    small_box(ax, Box(b4.x + 0.037, b4.y + 0.052, 0.335, 0.050), "异质性：产品类别 · 风险主题 · 公司差异 · 地区差异", fc="#FFFFFF", ec=PAL["orange2"], fontsize=6.2)

    thick_arrow(ax, b1.right, b2.left, PAL["orange"])
    thick_arrow(ax, b1.bottom, b3.top, PAL["orange"])
    thick_arrow(ax, b2.bottom, b4.top, PAL["orange"])
    thick_arrow(ax, b3.right, b4.left, PAL["orange"])
    add_footer(ax)
    save(fig, "mechanism_v1_quadrant_compact")


def version_b(profile):
    """Three-stage horizontal pipeline with compact subpanels."""
    fig, ax = make_ax(figsize=(9.4, 4.6))
    text(ax, 0.50, 0.940, "从投诉文本到企业响应：LLM 表达机制与实证分析流程", size=13, weight="bold")
    outer = Box(0.035, 0.080, 0.930, 0.820)
    gradient_rect(ax, outer, "#FFFFFF", "#F9FBFD", radius=0.018, ec="#AAB6C1", lw=1.1, z=0)

    stages = [
        (Box(0.060, 0.215, 0.245, 0.565), "1  数据与问题", PAL["blue_bg"], PAL["blue"]),
        (Box(0.335, 0.215, 0.300, 0.565), "2  表达与建模", PAL["purple_bg"], PAL["purple"]),
        (Box(0.665, 0.215, 0.275, 0.565), "3  响应与解释", PAL["green_bg"], PAL["green"]),
    ]
    for i, (box, title, bg, ec) in enumerate(stages):
        panel(ax, box, title, bg, "#FFFFFF", ec, chr(ord("a") + i))

    # stage 1
    b = stages[0][0]
    draw_documents_icon(ax, b.x + 0.035, b.y + 0.345, 1.18)
    small_box(ax, Box(b.x + 0.125, b.y + 0.400, 0.095, 0.052), "CFPB\n投诉叙事", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.3)
    small_box(ax, Box(b.x + 0.125, b.y + 0.330, 0.095, 0.052), "企业回应\n结果变量", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.3)
    small_box(ax, Box(b.x + 0.030, b.y + 0.215, 0.190, 0.055), f"{profile.get('rows', 30263):,} 条样本 · {profile.get('unique_products', 17)} 类产品", fc="#FFFFFF", ec=PAL["grid"], fontsize=6.1)
    small_box(ax, Box(b.x + 0.030, b.y + 0.125, 0.190, 0.057), "研究问题：LLM 是否改变\n金融投诉表达方式？", fc=PAL["blue_bg"], ec=PAL["blue"], fontsize=6.3)

    # stage 2
    b = stages[1][0]
    draw_chat_pen_icon(ax, b.x + 0.032, b.y + 0.382, 1.05)
    draw_scale_icon(ax, b.x + 0.155, b.y + 0.382, 0.95)
    draw_folder_icon(ax, b.x + 0.235, b.y + 0.382, 0.90)
    for i, lab in enumerate(["结构化", "法律化", "证据化", "模板化"]):
        small_box(ax, Box(b.x + 0.025 + i * 0.068, b.y + 0.300, 0.058, 0.043), lab, fc="#FFFFFF", ec=[PAL["blue"], PAL["purple"], PAL["yellow"], PAL["orange"]][i], fontsize=5.9)
    draw_matrix_icon(ax, b.x + 0.038, b.y + 0.185, 0.95)
    draw_topic_clusters(ax, b.x + 0.145, b.y + 0.180, 0.90)
    draw_network_icon(ax, b.x + 0.242, b.y + 0.178, 0.75)
    small_box(ax, Box(b.x + 0.038, b.y + 0.102, 0.075, 0.044), "文本\n特征", fc="#FFFFFF", ec=PAL["blue"], fontsize=5.8)
    small_box(ax, Box(b.x + 0.135, b.y + 0.102, 0.075, 0.044), "主题\n模型", fc="#FFFFFF", ec=PAL["teal"], fontsize=5.8)
    small_box(ax, Box(b.x + 0.232, b.y + 0.102, 0.075, 0.044), "预测\n模型", fc="#FFFFFF", ec=PAL["purple"], fontsize=5.8)

    # stage 3
    b = stages[2][0]
    draw_company_icon(ax, b.x + 0.035, b.y + 0.373, 1.05)
    draw_outcome_icons(ax, b.x + 0.154, b.y + 0.397, 1.20)
    for i, lab in enumerate(["金钱救济", "非金钱救济", "及时回应"]):
        small_box(ax, Box(b.x + 0.042 + i * 0.072, b.y + 0.305, 0.065, 0.040), lab, fc="#FFFFFF", ec=[PAL["orange"], PAL["green"], PAL["blue"]][i], fontsize=5.6)
    draw_barmini(ax, b.x + 0.048, b.y + 0.186, 1.25)
    small_box(ax, Box(b.x + 0.170, b.y + 0.178, 0.075, 0.055), "可解释\n信号", fc="#FFFFFF", ec=PAL["purple"], fontsize=6.0)
    small_box(ax, Box(b.x + 0.040, b.y + 0.105, 0.205, 0.047), "异质性：产品 · 主题 · 公司 · 地区", fc="#FFFFFF", ec=PAL["green"], fontsize=5.8)

    # Inter-stage arrows and lower feedback.
    thick_arrow(ax, stages[0][0].right, stages[1][0].left, PAL["orange"])
    thick_arrow(ax, stages[1][0].right, stages[2][0].left, PAL["orange"])
    arrow(ax, (0.800, 0.185), (0.205, 0.185), color=PAL["teal"], lw=1.3, rad=-0.12, ms=12, alpha=0.85)
    text(ax, 0.50, 0.153, "稳健性回路：时间比较、主题异质性、产品异质性与模型解释交叉验证", size=6.4, color=PAL["muted"])
    add_footer(ax)
    save(fig, "mechanism_v2_pipeline_board")


def version_c(profile):
    """Reference-like dense dashboard with a central method block and right evidence block."""
    fig, ax = make_ax(figsize=(9.2, 5.35))
    outer = Box(0.030, 0.055, 0.940, 0.890)
    gradient_rect(ax, outer, "#FFFFFF", "#FAFCFF", radius=0.012, ec="#A7B1BB", lw=1.0, z=0)

    top_left = Box(0.055, 0.675, 0.415, 0.205)
    top_right = Box(0.535, 0.675, 0.405, 0.205)
    main = Box(0.055, 0.265, 0.495, 0.345)
    right = Box(0.585, 0.265, 0.355, 0.345)
    bottom = Box(0.055, 0.105, 0.885, 0.105)

    panel(ax, top_left, "投诉数据与变量体系", PAL["blue_bg"], "#FFFFFF", PAL["blue"], "a")
    panel(ax, top_right, "LLM 辅助表达策略", PAL["green_bg"], "#FFFFFF", PAL["green"], "b")
    panel(ax, main, "文本机器学习分析框架", PAL["orange_bg"], "#FFFFFF", PAL["orange"], "c")
    panel(ax, right, "企业响应与机制解释", PAL["yellow_bg"], "#FFFFFF", PAL["yellow"], "d")
    panel(ax, bottom, "稳健性与异质性分析", PAL["purple_bg"], "#FFFFFF", PAL["purple"], "e")

    text(ax, 0.50, 0.940, "LLM 能否改变金融投诉表达方式？研究框架与机制路径", size=13, weight="bold")

    # Top-left
    draw_documents_icon(ax, top_left.x + 0.030, top_left.y + 0.055, 1.05)
    draw_database_icon(ax, top_left.x + 0.115, top_left.y + 0.062, 0.95)
    for i, lab in enumerate(["投诉叙事", "产品/问题", "公司回应", "日期/地区"]):
        small_box(ax, Box(top_left.x + 0.200 + (i % 2) * 0.100, top_left.y + 0.090 - (i // 2) * 0.055, 0.088, 0.043), lab, fc="#FFFFFF", ec=PAL["blue"], fontsize=5.8)
    small_box(ax, Box(top_left.x + 0.030, top_left.y + 0.020, 0.350, 0.032), f"样本：{profile.get('rows', 30263):,} 条叙事投诉，覆盖 {profile.get('unique_companies', 1371)} 家公司", fc="#FFFFFF", ec=PAL["grid"], fontsize=5.8)

    # Top-right
    draw_chat_pen_icon(ax, top_right.x + 0.030, top_right.y + 0.067, 1.10)
    draw_scale_icon(ax, top_right.x + 0.145, top_right.y + 0.070, 0.95)
    draw_folder_icon(ax, top_right.x + 0.245, top_right.y + 0.073, 0.90)
    for i, lab in enumerate(["结构化叙事", "法律化表达", "证据组织", "模板化短语"]):
        small_box(ax, Box(top_right.x + 0.035 + i * 0.087, top_right.y + 0.024, 0.075, 0.036), lab, fc="#FFFFFF", ec=[PAL["blue"], PAL["purple"], PAL["yellow"], PAL["orange"]][i], fontsize=5.3)

    # Main method block
    small_box(ax, Box(main.x + 0.030, main.y + 0.225, 0.113, 0.060), "文本清洗\n与标准化", fc="#FFFFFF", ec=PAL["blue"], fontsize=6.0)
    small_box(ax, Box(main.x + 0.180, main.y + 0.225, 0.113, 0.060), "表达特征\n构造", fc="#FFFFFF", ec=PAL["purple"], fontsize=6.0)
    small_box(ax, Box(main.x + 0.330, main.y + 0.225, 0.113, 0.060), "主题建模\nNMF/BERTopic", fc="#FFFFFF", ec=PAL["teal"], fontsize=6.0)
    thick_arrow(ax, (main.x + 0.145, main.y + 0.255), (main.x + 0.176, main.y + 0.255), PAL["orange"])
    thick_arrow(ax, (main.x + 0.295, main.y + 0.255), (main.x + 0.326, main.y + 0.255), PAL["orange"])
    draw_matrix_icon(ax, main.x + 0.050, main.y + 0.112, 0.95)
    draw_topic_clusters(ax, main.x + 0.205, main.y + 0.105, 1.0)
    draw_network_icon(ax, main.x + 0.365, main.y + 0.105, 0.90)
    small_box(ax, Box(main.x + 0.030, main.y + 0.030, 0.415, 0.043), "特征输入：词频、文本长度、法律词、证据词、主题权重、产品与地区变量", fc="#FFFFFF", ec=PAL["orange2"], fontsize=5.7)

    # Right evidence block
    draw_company_icon(ax, right.x + 0.035, right.y + 0.223, 1.0)
    draw_outcome_icons(ax, right.x + 0.145, right.y + 0.248, 1.1)
    draw_barmini(ax, right.x + 0.245, right.y + 0.225, 1.05)
    for i, lab in enumerate(["救济响应预测", "可解释特征排序", "机制假设验证"]):
        small_box(ax, Box(right.x + 0.040, right.y + 0.135 - i * 0.050, 0.270, 0.037), lab, fc="#FFFFFF", ec=[PAL["green"], PAL["purple"], PAL["orange"]][i], fontsize=5.9)

    # Bottom
    for i, lab in enumerate(["ChatGPT 前后", "主题异质性", "产品异质性", "公司/地区差异", "稳健性检验"]):
        small_box(ax, Box(bottom.x + 0.035 + i * 0.167, bottom.y + 0.028, 0.125, 0.044), lab, fc="#FFFFFF", ec=[PAL["purple"], PAL["teal"], PAL["blue"], PAL["green"], PAL["orange"]][i], fontsize=5.8)
        if i < 4:
            arrow(ax, (bottom.x + 0.160 + i * 0.167, bottom.y + 0.050), (bottom.x + 0.192 + i * 0.167, bottom.y + 0.050), color=PAL["line"], lw=1.0, ms=10)

    thick_arrow(ax, top_left.right, top_right.left, PAL["orange"])
    thick_arrow(ax, top_left.bottom, main.top, PAL["orange"])
    thick_arrow(ax, top_right.bottom, right.top, PAL["orange"])
    thick_arrow(ax, main.right, right.left, PAL["orange"])
    arrow(ax, right.bottom, (bottom.x + bottom.w * 0.72, bottom.y + bottom.h), color=PAL["purple"], lw=1.2, rad=0.10, ms=12)
    add_footer(ax)
    save(fig, "mechanism_v3_dashboard_dense")


def main() -> None:
    configure_style()
    profile = load_profile()
    version_a(profile)
    version_b(profile)
    version_c(profile)


if __name__ == "__main__":
    main()
