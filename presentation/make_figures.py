import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#d9d8d3", "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "font.size": 11,
})

def tidy(ax, ymax=1.0):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#eceae5", lw=1)
    ax.xaxis.grid(False)
    ax.spines["left"].set_color("#d9d8d3")
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.set_ylim(0, ymax)
    ax.tick_params(length=0)

def grouped(path, labels, s1, s2, n1, n2, figsize=(11.2, 4.5), note=None, divider=None):
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    b1 = ax.bar(x - w/2 - 0.012, s1, w, label=n1, color=BLUE)
    b2 = ax.bar(x + w/2 + 0.012, s2, w, label=n2, color=ORANGE)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("f1 @ IoU 0.5", fontsize=10)
    tidy(ax, max(max(s1), max(s2)) * 1.32)
    if divider is not None:
        ax.axvline(divider, color="#d9d8d3", lw=1, ls=(0, (3, 3)))
    leg = ax.legend(frameon=False, loc="upper right", fontsize=10, ncols=2)
    for t in leg.get_texts(): t.set_color(INK2)
    if note:
        ax.text(0, -0.20, note, transform=ax.transAxes, fontsize=9,
                color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(OUT / path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

# --- 1. backbone selection (phase 1, n=150) -------------------------------
types = ["PLAIN", "ORDINAL", "BEFORE", "AFTER", "WHILE", "NEXT_AFTER", "NOT_FOLLOWED"]
omni  = [0.375, 0.333, 0.379, 0.146, 0.152, 0.091, 0.095]
audio = [0.094, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
grouped("fig1_backbone.png", types, omni, audio,
        "Qwen2.5-Omni-7B", "Qwen2-Audio-7B-Instruct",
        note="Composed ESC-50, 150 queries per model, zero-shot, 4-bit on a T4.")

# --- 2. THE headline: direct vs decomposition, test split -----------------
order = ["NEXT_AFTER", "NOT_FOLLOWED", "ORDINAL", "AFTER", "PLAIN", "WHILE", "BEFORE"]
A = {"PLAIN":0.276,"ORDINAL":0.188,"AFTER":0.220,"BEFORE":0.323,
     "NEXT_AFTER":0.101,"WHILE":0.175,"NOT_FOLLOWED":0.146}
B = {"PLAIN":0.295,"ORDINAL":0.212,"AFTER":0.234,"BEFORE":0.228,
     "NEXT_AFTER":0.159,"WHILE":0.146,"NOT_FOLLOWED":0.215}
grouped("fig2_direct_vs_decompose.png", order,
        [A[t] for t in order], [B[t] for t in order],
        "Direct prompting", "Decompose-and-combine", divider=4.5,
        note="Qwen2.5-Omni-7B, held-out test split, 699 queries. Left of the dashed line: "
             "decomposition predicted to win. Right: predicted to lose.")

# --- 3. the asymmetry -----------------------------------------------------
rate = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
miss = [0.913, 0.844, 0.780, 0.712, 0.650, 0.590]
fa   = [0.987, 0.975, 0.961, 0.953, 0.937, 0.923]
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
ax.plot(rate, miss, color=BLUE, lw=2, marker="o", ms=7, label="missed events")
ax.plot(rate, fa, color=ORANGE, lw=2, marker="o", ms=7, label="false detections")
ax.text(0.302, 0.590, "  slope 1.411", color=BLUE, fontsize=10, va="center")
ax.text(0.302, 0.923, "  slope 0.252", color=ORANGE, fontsize=10, va="center")
ax.set_xlabel("error rate injected into a perfect grounder", fontsize=10)
ax.set_ylabel("f1 @ IoU 0.5", fontsize=10)
ax.set_xlim(0.03, 0.40); ax.set_xticks(rate)
tidy(ax, 1.04); ax.set_ylim(0.5, 1.02)
leg = ax.legend(frameon=False, loc="lower left", fontsize=10)
for t in leg.get_texts(): t.set_color(INK2)
ax.text(0, -0.22, "Full ESC-50 benchmark, 4404 queries. One failure mode degraded at a time.",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
fig.tight_layout(); fig.savefig(OUT / "fig3_asymmetry.png", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# --- 4. the negative result ----------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=200)
names = ["Single sample\n(k=1)", "Vote-and-merge\n(k=3, 2 votes)"]
vals = [0.207, 0.150]
bars = ax.bar(names, vals, 0.42, color=[BLUE, ORANGE])
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center",
            va="bottom", fontsize=12, color=INK)
ax.axhline(0.951, color=MUTED, lw=1.2, ls=(0, (4, 3)))
ax.text(1.42, 0.951, "0.951 — what the simulation predicted", fontsize=10,
        color=MUTED, va="center", ha="right")
ax.set_ylabel("f1 @ IoU 0.5, all 699 test queries", fontsize=10)
tidy(ax, 1.08)
ax.text(0, -0.16, "Real sampling errors are correlated: the same event is mislocated the same way "
        "in every sample,\nso voting discards real events instead of filtering spurious ones. "
        "Under-report rate rose 0.177 → 0.538.",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
fig.tight_layout(); fig.savefig(OUT / "fig4_decoding.png", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))


# --- revisions applied after review (override figs 2 and 4) ---
import numpy as np, matplotlib.pyplot as plt

A = {"PLAIN":0.276,"ORDINAL":0.188,"AFTER":0.220,"BEFORE":0.323,
     "NEXT_AFTER":0.101,"WHILE":0.175,"NOT_FOLLOWED":0.146}
B = {"PLAIN":0.295,"ORDINAL":0.212,"AFTER":0.234,"BEFORE":0.228,
     "NEXT_AFTER":0.159,"WHILE":0.146,"NOT_FOLLOWED":0.215}
# grouped by what the CPU study PREDICTED, marked with what actually happened
pred_decomp = ["NEXT_AFTER", "NOT_FOLLOWED", "AFTER", "WHILE"]
pred_direct = ["ORDINAL", "BEFORE"]
order = pred_decomp + pred_direct
held = {t: (B[t] > A[t]) == (t in pred_decomp) for t in order}

x = np.arange(len(order)); w = 0.36
fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=200)
s1 = [A[t] for t in order]; s2 = [B[t] for t in order]
b1 = ax.bar(x - w/2 - 0.012, s1, w, label="Direct prompting", color=BLUE)
b2 = ax.bar(x + w/2 + 0.012, s2, w, label="Decompose-and-combine", color=ORANGE)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
ax.set_ylabel("f1 @ IoU 0.5", fontsize=10)
tidy(ax, 0.44)
ax.axvline(3.5, color="#d9d8d3", lw=1, ls=(0, (3, 3)))
ax.text(1.5, 0.415, "predicted: decomposition wins", ha="center", fontsize=9.5, color=MUTED)
ax.text(4.5, 0.415, "predicted: direct wins", ha="center", fontsize=9.5, color=MUTED)
for i, t in enumerate(order):
    ax.text(i, -0.052, "held" if held[t] else "flipped", ha="center", fontsize=9,
            color=BLUE if held[t] else "#b14a2a",
            fontweight="bold" if not held[t] else "normal")
leg = ax.legend(frameon=False, loc="upper center", fontsize=10, ncols=2,
                bbox_to_anchor=(0.5, 1.14))
for t in leg.get_texts(): t.set_color(INK2)
ax.text(0, -0.20, "Qwen2.5-Omni-7B, held-out test split, 699 queries.  Four of the six "
        "per-type predictions held; WHILE and ORDINAL flipped.",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
fig.tight_layout(); fig.savefig(OUT/"fig2_direct_vs_decompose.png", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# fig 4 redo
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
names = ["Single sample\n(k=1)", "Vote-and-merge\n(k=3, 2 votes)"]
vals = [0.207, 0.150]
bars = ax.bar([0, 0.62], vals, 0.30, color=[BLUE, ORANGE])
ax.set_xticks([0, 0.62]); ax.set_xticklabels(names, fontsize=11)
ax.set_xlim(-0.42, 1.04)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center",
            va="bottom", fontsize=13, color=INK)
ax.axhline(0.951, color=MUTED, lw=1.2, ls=(0, (4, 3)), xmax=0.62)
ax.text(0.64, 0.951, "0.951\nwhat the simulation\npredicted", fontsize=9.5,
        color=MUTED, va="center", ha="left", linespacing=1.45)
ax.set_ylabel("f1 @ IoU 0.5, all 699 test queries", fontsize=10)
tidy(ax, 1.10)
ax.text(0, -0.17, "Real sampling errors are correlated: the same event is mislocated the same way\n"
        "in every sample, so voting discards real events rather than filtering spurious ones.\n"
        "Under-report rate rose 0.177 → 0.538.",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT/"fig4_decoding.png", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)
print("held/flipped:", held)

