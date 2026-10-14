#!/usr/bin/env python3
"""
Publication architecture figures for the 2D ADMM-PINN study (v4, styled).

Same deterministic, code-accurate schematics as v3, restyled for a clean,
modern, attractive look:
  * rounded boxes with soft drop shadows
  * colour-coded stages (input/output, encoder, bottleneck, decoder,
    head/refine, attention gates, loss terms)
  * refined arrows and a compact colour legend
  * uniform grid -> identical box sizes and identical forward-arrow lengths
Matches the code exactly (2-channel input, 64x128 patches, 16/32/64/128
U-Net channels, prediction L_pred = L_prior + scale*residual + bias with
scale, bias learnable and initialised 0.1 / 0.0).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parent / "ARCHITECTURE_FIGURES"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---- palette -------------------------------------------------------------
INK = "#1f2937"          # primary text
EDGE = "#334155"         # box outline
ARROW = "#334155"        # forward arrows
SKIP = "#64748b"         # skip / dashed
SHADOW = "#00000016"     # soft shadow

C_IO = "#eef2f7"         # input / output
C_ENC = ["#e3edff", "#c8dbff", "#a8c6fb"]   # Enc1, Enc2, Enc3
C_BOTT = "#3b4cca"       # bottleneck (white text)
C_DEC = ["#a9ddcb", "#c2e7d8", "#dcf3ec"]   # Dec3, Dec2, Dec1 (deep->light)
C_HEAD = "#ffe1b0"       # 1x1 head / refine
C_AG = "#ffd3dd"         # attention gates
C_LOSS = "#dcf5e3"       # loss terms
C_CORE = "#dbe5fb"       # ADMM / refiner core (Fig 1)

MUL = "×"
ELL1 = r"$\mathbf{\ell_{1}}$"   # mathtext (STIX, Times-like); Times New Roman lacks U+2113

plt.rcParams.update({
    "font.weight": "bold",
    "mathtext.default": "bf",
    "font.size": 8.7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- uniform grid (architecture panels) ----------------------------------
X0, P, W, H = 0.30, 2.00, 1.60, 0.92
YB = 1.05
YT = YB + H
YCEN = YB + H / 2
BOX_FS = 7.9
NOTE_FS = 8.3
TITLE_FS = 11.5
XLIM, YLIM = 20.0, 4.80
ROUND = "round,pad=0,rounding_size=0.10"


def box(ax, x, y, w, h, label, fill, tc=INK, fs=BOX_FS, lw=1.0, shadow=True, edge=EDGE):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.06), w, h, boxstyle=ROUND,
                                    facecolor=SHADOW, edgecolor="none",
                                    mutation_aspect=1.0, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=ROUND,
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                mutation_aspect=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold", linespacing=1.18, zorder=3)


def arrow(ax, x1, y1, x2, y2, style="solid", lw=1.1, mutation=11, color=ARROW):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=mutation, linewidth=lw,
                                 linestyle=style, color=color,
                                 shrinkA=0, shrinkB=0, zorder=2.5))


def cx(i):
    return X0 + i * P + W / 2


def row(ax, items):
    for i, (label, fill, tc) in enumerate(items):
        box(ax, X0 + i * P, YB, W, H, label, fill, tc=tc)
    for i in range(len(items) - 1):
        arrow(ax, X0 + i * P + W, YCEN, X0 + (i + 1) * P, YCEN)


def skip_arc(ax, a, b, yh, label="skip", dest_top=YT):
    xa, xb = cx(a), cx(b)
    ax.add_line(Line2D([xa, xa, xb], [YT, yh, yh], color=SKIP,
                       linewidth=1.0, linestyle=(0, (4, 2)), zorder=1.5))
    arrow(ax, xb, yh, xb, dest_top, style="dashed", lw=1.0, color=SKIP)
    ax.text((xa + xb) / 2, yh + 0.05, label, ha="center", va="bottom",
            fontsize=7.4, color=SKIP, style="italic", fontweight="bold")


def identity_bump(ax, i, yh=2.55):
    xl = X0 + i * P + 0.28
    xr = X0 + i * P + W - 0.28
    ax.add_line(Line2D([xl, xl, xr], [YT, yh, yh], color=SKIP,
                       linewidth=1.0, zorder=1.5))
    arrow(ax, xr, yh, xr, YT, lw=1.0, mutation=9, color=SKIP)
    ax.text((xl + xr) / 2, yh + 0.04, "identity", ha="center", va="bottom",
            fontsize=7.2, color=SKIP, style="italic", fontweight="bold")


def frame(ax, title):
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.axis("off")
    ax.text(0.0, YLIM - 0.28, title, fontsize=TITLE_FS, fontweight="bold",
            ha="left", color=INK)


# ---- Figure 2 panels -----------------------------------------------------
def unet(ax):
    frame(ax, "(a) 2D U-Net")
    items = [
        (f"Input\n2 {MUL} 64 {MUL} 128", C_IO, INK),
        (f"Enc1\n16 {MUL} 64 {MUL} 128", C_ENC[0], INK),
        (f"Enc2\n32 {MUL} 32 {MUL} 64", C_ENC[1], INK),
        (f"Enc3\n64 {MUL} 16 {MUL} 32", C_ENC[2], INK),
        (f"Bottleneck\n128 {MUL} 8 {MUL} 16", C_BOTT, "white"),
        (f"Dec3\n64 {MUL} 16 {MUL} 32", C_DEC[0], INK),
        (f"Dec2\n32 {MUL} 32 {MUL} 64", C_DEC[1], INK),
        (f"Dec1\n16 {MUL} 64 {MUL} 128", C_DEC[2], INK),
        (f"1 {MUL} 1 head\n(residual)", C_HEAD, INK),
        ("Output\nL = log Z", C_IO, INK),
    ]
    row(ax, items)
    skip_arc(ax, 3, 5, 2.55)
    skip_arc(ax, 2, 6, 3.00)
    skip_arc(ax, 1, 7, 3.45)
    ax.text(0.0, 0.58,
            f"Prediction rule:   L_pred = L_prior + scale {MUL} residual + bias    "
            "(scale, bias learnable; initialised 0.1, 0.0)",
            fontsize=NOTE_FS, ha="left", color=INK, fontweight="bold")
    ax.text(0.0, 0.18,
            f"L_prior is the ADMM log-impedance prior; the loss combines physics, "
            f"RW-{ELL1}, prior, lateral, supervised, and gradient terms.",
            fontsize=NOTE_FS, ha="left", color=INK, fontweight="bold")


def rescnn(ax):
    frame(ax, "(b) ResCNN")
    items = [
        (f"Input\n2 {MUL} 64 {MUL} 128", C_IO, INK),
        (f"Stem\n(3 {MUL} 7) conv\n16 ch", C_ENC[0], INK),
        ("ResBlock\nd = 1", C_ENC[1], INK),
        ("ResBlock\nd = 2", C_ENC[2], INK),
        ("ResBlock\nd = 4", C_ENC[2], INK),
        ("ResBlock\nd = 2", C_DEC[0], INK),
        ("ResBlock\nd = 1", C_DEC[1], INK),
        (f"Head\n(3 {MUL} 7) + 1 {MUL} 1", C_HEAD, INK),
        ("Output\nL = log Z", C_IO, INK),
    ]
    row(ax, items)
    for i in range(2, 7):
        identity_bump(ax, i)
    ax.text(0.0, 0.58,
            f"All residual blocks keep full 64 {MUL} 128 resolution and use "
            f"anisotropic (3 {MUL} 7) kernels.",
            fontsize=NOTE_FS, ha="left", color=INK, fontweight="bold")
    ax.text(0.0, 0.18,
            "Dilations 1-2-4-2-1 widen the receptive field; the output is added "
            "to the ADMM prior.",
            fontsize=NOTE_FS, ha="left", color=INK, fontweight="bold")


def attention(ax):
    frame(ax, "(c) Attention ResUNet")
    items = [
        (f"Input\n2 {MUL} 64 {MUL} 128", C_IO, INK),
        (f"Enc1\n16 {MUL} 64 {MUL} 128", C_ENC[0], INK),
        (f"Enc2\n32 {MUL} 32 {MUL} 64", C_ENC[1], INK),
        (f"Enc3\n64 {MUL} 16 {MUL} 32", C_ENC[2], INK),
        (f"Bottleneck\n128 {MUL} 8 {MUL} 16", C_BOTT, "white"),
        (f"Dec3\n64 {MUL} 16 {MUL} 32", C_DEC[0], INK),
        (f"Dec2\n32 {MUL} 32 {MUL} 64", C_DEC[1], INK),
        (f"Dec1\n16 {MUL} 64 {MUL} 128", C_DEC[2], INK),
        ("Refine\nResBlock", C_HEAD, INK),
        ("Output\nL = log Z", C_IO, INK),
    ]
    row(ax, items)
    ag_w, ag_h, ag_b = 0.96, 0.50, 2.20
    ag_t = ag_b + ag_h
    for dec_i, name in [(7, "AG1"), (6, "AG2"), (5, "AG3")]:
        box(ax, cx(dec_i) - ag_w / 2, ag_b, ag_w, ag_h, name, C_AG, fs=7.6)
        arrow(ax, cx(dec_i), ag_b, cx(dec_i), YT, style="dashed", lw=1.0, color=SKIP)
    skip_arc(ax, 3, 5, 3.00, dest_top=ag_t)
    skip_arc(ax, 2, 6, 3.45, dest_top=ag_t)
    skip_arc(ax, 1, 7, 3.90, dest_top=ag_t)
    ax.text(0.0, 0.58,
            "Attention gate:   alpha = sigmoid(psi(ReLU(theta(skip) + phi(gate))))",
            fontsize=NOTE_FS, ha="left", color=INK, fontweight="bold")
    ax.text(0.0, 0.18,
            "Each gate modulates the encoder skip using the decoder (gating) features "
            "before concatenation.",
            fontsize=NOTE_FS, ha="left", color=INK, fontweight="bold")


def draw_architecture():
    fig = plt.figure(figsize=(13.0, 11.3))
    gs = fig.add_gridspec(3, 1, hspace=0.34, top=0.965, bottom=0.075,
                          left=0.015, right=0.985)
    unet(fig.add_subplot(gs[0, 0]))
    rescnn(fig.add_subplot(gs[1, 0]))
    attention(fig.add_subplot(gs[2, 0]))

    legend_items = [
        ("Input / Output", C_IO), ("Encoder", C_ENC[1]),
        ("Bottleneck", C_BOTT), ("Decoder", C_DEC[1]),
        ("Head / Refine", C_HEAD), ("Attention gate", C_AG),
    ]
    handles = [Patch(facecolor=f, edgecolor=EDGE, label=l) for l, f in legend_items]
    leg = fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                     fontsize=9.5, handlelength=1.3, columnspacing=1.6,
                     bbox_to_anchor=(0.5, 0.012))
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.savefig(OUT_DIR / "2D_Fig15_Deep_Learning_Architectures.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "2D_Fig15_Deep_Learning_Architectures.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 1 : workflow (restyled to match) -----------------------------
def draw_workflow():
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    yc, Hm = 3.20, 0.92
    ym = yc - Hm / 2
    yt = yc + 0.55 - 0.31
    yb = yc - 0.55 - 0.31
    hs = 0.62

    box(ax, 0.25, yt, 1.55, hs, "Seismic\nsection S", C_IO, fs=8.0)
    box(ax, 0.25, yb, 1.55, hs, "Initial\nmodel L0", C_IO, fs=8.0)
    box(ax, 2.30, ym, 1.95, Hm, f"Trace-wise\nreweighted {ELL1}\nADMM (prior)", C_CORE, fs=8.0)
    box(ax, 4.85, yt, 1.65, hs, "ADMM prior\nL_ADMM", C_IO, fs=8.0)
    box(ax, 4.85, yb, 1.65, hs, "Seismic patch\nS_patch", C_IO, fs=8.0)
    box(ax, 7.05, ym, 1.95, Hm, "2D neural refiner\nU-Net / ResCNN /\nAttention ResUNet", C_BOTT, tc="white", fs=8.0)
    box(ax, 9.75, yc - 0.39, 1.75, 0.78, "Predicted log\nimpedance L", C_DEC[1], fs=8.0)
    box(ax, 11.85, ym, 1.75, Hm, "Acoustic\nimpedance\nZ = exp(L)", C_HEAD, fs=8.0)

    yl, hl = 0.72, 0.62
    losses = [(3.95, "prior\nproximity"), (5.40, "physics\ndata term"),
              (6.85, f"RW-{ELL1}\nsparsity"), (8.30, "lateral\nsmoothness"),
              (9.75, "synthetic sup.\n+ gradient")]
    for lx, lab in losses:
        wlx = 1.50 if "synthetic" in lab else 1.30
        box(ax, lx, yl, wlx, hl, lab, C_LOSS, fs=8.0)

    arrow(ax, 1.80, yt + hs / 2, 2.30, yc + 0.18)
    arrow(ax, 1.80, yb + hs / 2, 2.30, yc - 0.18)
    arrow(ax, 4.25, yc, 4.85, yt + hs / 2)
    arrow(ax, 6.50, yt + hs / 2, 7.05, yc + 0.18)
    arrow(ax, 6.50, yb + hs / 2, 7.05, yc - 0.18)
    arrow(ax, 9.00, yc, 9.75, yc)
    arrow(ax, 11.50, yc, 11.85, yc)

    src_x, src_y = 9.75 + 1.75 / 2, yc - 0.39
    for cxl in (4.60, 6.05, 7.50, 8.95, 10.50):
        arrow(ax, src_x, src_y, cxl, yl + hl, style="dashed", lw=0.9, color=SKIP)

    fig.savefig(OUT_DIR / "2D_Fig14_ADMM_Guided_Neural_Workflow.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "2D_Fig14_ADMM_Guided_Neural_Workflow.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_workflow()
    draw_architecture()
    print(OUT_DIR / "2D_Fig14_ADMM_Guided_Neural_Workflow.png")
    print(OUT_DIR / "2D_Fig15_Deep_Learning_Architectures.png")
