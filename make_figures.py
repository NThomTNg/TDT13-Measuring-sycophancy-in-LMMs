"""Generate the figures for the report: scatter.{pdf,png} and pipeline.{pdf,png}."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PROP, OPEN = "#2a78d6", "#eb6834"  
BOX_FILL = "#f3f2ef"


# ---------------------------------------------------------------------------
# Figure 1: Turn of Flip (SYCON-Bench) vs. S^moral (ELEPHANT)
# ---------------------------------------------------------------------------
# (name, turn_of_flip, s_moral, group, match_level, label_offset, label_ha)
MODELS = [
    ("Llama-3-8B / 3.1-8B", 1.45, .68, "open", "L2", (0, 11),  "center"),
    ("Llama-3.3-70B",       1.90, .67, "open", "L1", (-4, 11), "right"),
    ("Qwen2.5-7B",          1.93, .62, "open", "L1", (4, -12), "left"),
    ("DeepSeek-V3",         2.88, .65, "open", "L2", (-9, 0),  "right"),
    ("GPT-4o",              2.92, .40, "prop", "L2", (-9, 0),  "right"),
    ("Claude 3.7 Sonnet",   2.92, .15, "prop", "L1", (-9, 0),  "right"),
]
S_MORAL_ERR = 0.04


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)


def scatter_legend_handles():
    def marker(shape, fill, edge):
        return dict(marker=shape, ls="", mfc=fill, mec=edge, ms=6)

    return [
        Line2D([], [], **marker("s", PROP, PROP), label="Proprietary"),
        Line2D([], [], **marker("o", OPEN, OPEN), label="Open-weight"),
        Line2D([], [], **marker("o", INK2, INK2), label="L1 (exact)"),
        Line2D([], [], **marker("o", "white", INK2), label="L2 (same line)"),
    ]


def make_scatter():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))

    for name, tof, s, group, level, offset, ha in MODELS:
        color = PROP if group == "prop" else OPEN
        face = color if level == "L1" else "white"
        ax.errorbar(tof, s, yerr=S_MORAL_ERR, fmt="none", ecolor=color,
                    elinewidth=1, capsize=0, alpha=.6, zorder=2)
        ax.scatter(tof, s, s=46, marker="o" if group == "open" else "s",
                   facecolor=face, edgecolor=color, linewidth=1.6, zorder=3)
        ax.annotate(name, (tof, s), xytext=offset, textcoords="offset points",
                    ha=ha, va="center", fontsize=8, color=INK)

    ax.set_xlim(3.75, 1.2)  
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Turn of Flip, false presuppositions (SYCON-Bench)\n"
                  "$\\leftarrow$ holds longer      flips sooner $\\rightarrow$", color=INK2)
    ax.set_ylabel("$S^{\\mathrm{moral}}$ (ELEPHANT)", color=INK2)
    style_axes(ax)
    ax.legend(handles=scatter_legend_handles(), loc="lower right",
              fontsize=7.5, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig("scatter.pdf")
    fig.savefig("scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: method pipeline diagram
# ---------------------------------------------------------------------------
STEPS = ["Select\nstudies", "Extract\nmodel scores",
         "Match models\n(L1 / L2 / L3)", "Rank models\nin each study"]


def draw_box(ax, x, y, w, h, text, ec=INK2, fc=BOX_FILL, lw=.8, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=INK)


def draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=.9, shrinkA=0, shrinkB=0))


def make_pipeline():
    fig, ax = plt.subplots(figsize=(6.4, 1.7))
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 34)
    ax.axis("off")

    # Main row of steps
    w, h, y, gap = 21, 12, 17, 5
    xs = [1 + k*(w + gap) for k in range(len(STEPS))]
    for x, text in zip(xs, STEPS):
        draw_box(ax, x, y, w, h, text)
    for x in xs[:-1]:
        draw_arrow(ax, x + w + .8, y + h/2, x + w + gap - .8, y + h/2)

    # Branch from the last step into the two ranking analyses
    last_right = xs[-1] + w + .8
    ox = xs[-1] + w + gap + 1
    draw_box(ax, ox, 24, 20, 8.5, "Across types\n(RQ2)", ec=PROP, fc="white", lw=1.2, fs=7.2)
    draw_box(ax, ox, 13.5, 20, 8.5, "Within a type\n(RQ3)", ec=OPEN, fc="white", lw=1.2, fs=7.2)
    draw_arrow(ax, last_right, y + h/2 + 1, ox - .8, 28.2)
    draw_arrow(ax, last_right, y + h/2 - 1, ox - .8, 17.7)

    # Coverage analysis hangs off the matching step
    mx = xs[2] + w/2
    draw_arrow(ax, mx, y - .8, mx, 7.5)
    ax.text(mx, 4.5, "Coverage of each study (RQ1)", ha="center", va="center",
            fontsize=7.4, color=INK2, style="italic")

    save_kw = dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig("pipeline.pdf", **save_kw)
    fig.savefig("pipeline.png", dpi=150, **save_kw)
    plt.close(fig)


if __name__ == "__main__":
    make_scatter()
    make_pipeline()
