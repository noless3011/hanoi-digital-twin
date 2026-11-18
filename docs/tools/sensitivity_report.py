"""Builds docs/parameter-sensitivity.pdf from runs/twin/sensitivity-{choice,priority}/sensitivity.json.

Charts are matplotlib (saved as PNG at 200 dpi), the document is reportlab.

    python docs/tools/sensitivity_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "docs" / "parameter-sensitivity.pdf"
FIG_DIR = ROOT / "runs" / "twin" / "sensitivity-figures"
RUNS = {"choice": ROOT / "runs/twin/sensitivity-choice/sensitivity.json",
        "priority": ROOT / "runs/twin/sensitivity-priority/sensitivity.json"}

# Reference palette (dataviz skill): light surface, ink, sequential blue, diverging blue <-> red.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#c9c8c2"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PARTS = ["motion", "spacing", "occupancy_map", "occupancy_time", "crossings", "headways"]
PART_LABEL = {"motion": "motion", "spacing": "spacing", "occupancy_map": "occupancy\nmap",
              "occupancy_time": "occupancy\nover time", "crossings": "crossings", "headways": "headways"}

for f in ("segoeui.ttf", "seguisb.ttf"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "xtick.major.size": 0,
    "ytick.major.size": 0, "axes.titlelocation": "left", "axes.titlepad": 8,
})


def label(name):
    """Readable parameter names; speed-profile knots become 'speed route r @ k/3'."""
    if name.startswith("v_r") and "_k" in name:
        r, k = name[3:].split("_k")
        return f"speed, route {r}, point {k}/3"
    return name


def rounded_barh(ax, y, w, color, height=0.62):
    bars = ax.barh(y, w, height=height, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=2)
    for b in bars:
        b.set_capstyle("round")
    return bars


def fig_effect(rep, metric, title, xlabel, path, top=None):
    """Horizontal bars of mean |change| per parameter, the seed-noise floor as a dashed line."""
    eff = rep["effects"]
    noise = rep["noise_floor"][metric]
    names = sorted(rep["names"], key=lambda n: eff[n][metric]["effect"])
    if top:
        names = names[-top:]
    vals = np.array([eff[n][metric]["effect"] for n in names])
    colors = [BLUE if v > 2 * noise else (NEUTRAL if v <= noise else "#9ec5f4") for v in vals]
    h = 0.24 * len(names) + 0.9
    fig, ax = plt.subplots(figsize=(6.6, h))
    y = np.arange(len(names))
    rounded_barh(ax, y, vals, colors)
    ax.axvline(noise, color=INK2, lw=1, ls=(0, (3, 2)), zorder=3)
    ax.axvline(2 * noise, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=3)
    ax.text(noise, len(names) - 0.35, " seed noise", color=INK2, fontsize=7.5, va="bottom")
    ax.text(2 * noise, len(names) - 0.35, " 2 × noise", color=MUTED, fontsize=7.5, va="bottom")
    ax.set_yticks(y, [label(n) for n in names])
    for yi, v in zip(y, vals):
        ax.text(v + vals.max() * 0.01, yi, f"{v:.3f}", va="center", fontsize=7.2, color=INK2)
    ax.set_xlim(0, max(vals.max(), 2 * noise) * 1.15)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="above 2 × noise"), Patch(color="#9ec5f4", label="1–2 × noise"),
                       Patch(color=NEUTRAL, label="within noise")], loc="lower right", frameon=False,
              fontsize=7.2, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_direction(rep, path):
    """Signed mean change of the score when the parameter goes up (diverging: blue lowers = closer)."""
    eff = rep["effects"]
    names = sorted(rep["names"], key=lambda n: eff[n]["score"]["effect"])
    vals = np.array([eff[n]["score"]["mean"] for n in names])
    sig = np.array([eff[n]["score"]["sigma"] for n in names])
    fig, ax = plt.subplots(figsize=(6.6, 0.24 * len(names) + 0.9))
    y = np.arange(len(names))
    rounded_barh(ax, y, vals, [BLUE if v < 0 else RED for v in vals])
    ax.errorbar(vals, y, xerr=sig, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=4)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [label(n) for n in names])
    lim = np.max(np.abs(vals) + sig) * 1.1
    ax.set_xlim(-lim, lim)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("mean change of the score when the parameter goes up (bar) ± σ (line)")
    ax.set_title("Direction: does raising the parameter bring the simulation closer to the video?")
    ax.text(-lim * 0.98, len(names) - 0.3, "← raising it brings the sim closer", color=INK2, fontsize=7.3, va="bottom")
    ax.text(lim * 0.98, len(names) - 0.3, "raising it moves the sim away →", color=INK2, fontsize=7.3, va="bottom",
            ha="right")
    ax.set_ylim(-0.6, len(names) + 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_parts(rep, path):
    """Heatmap: effect of each parameter on each score part (sequential blue), noise-level cells muted."""
    eff = rep["effects"]
    names = sorted(rep["names"], key=lambda n: -eff[n]["score"]["effect"])
    parts = [p for p in PARTS if f"part:{p}" in rep["noise_floor"]]
    M = np.array([[eff[n][f"part:{p}"]["effect"] for p in parts] for n in names])
    noise = np.array([rep["noise_floor"][f"part:{p}"] for p in parts])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(6.6, 0.26 * len(names) + 1.3))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=M.max())
    for i in range(len(names)):
        for j in range(len(parts)):
            v = M[i, j]
            weak = v <= noise[j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.8,
                    color=(MUTED if weak else ("white" if v > M.max() * 0.55 else INK)))
    ax.set_xticks(range(len(parts)), [PART_LABEL[p] for p in parts])
    ax.set_yticks(range(len(names)), [label(n) for n in names])
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(parts)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("mean |change| of the part", color=INK2)
    noise_txt = "  ".join(f"{PART_LABEL[p].replace(chr(10), ' ')} {n:.3f}" for p, n in zip(parts, noise))
    ax.set_title("Which part of the score each parameter moves", pad=28)
    fig.text(0.01, 0.005, "grey numbers: at or below that part's seed noise (" + noise_txt + ")",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_plane(rep, path):
    """Morris plane: effect (mu*) against sigma; labels on the parameters that matter."""
    eff = rep["effects"]
    names = rep["names"]
    mu = np.array([eff[n]["score"]["effect"] for n in names])
    sg = np.array([eff[n]["score"]["sigma"] for n in names])
    noise = rep["noise_floor"]["score"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axvspan(0, noise, color="#f0efec", zorder=0)
    ax.text(noise * 0.5, sg.max() * 1.08, "noise", ha="center", fontsize=7.5, color=MUTED)
    xmax, ymax = mu.max() * 1.3, sg.max() * 1.25
    d = min(xmax, ymax)
    ax.plot([0, d], [0, d], color=GRID, lw=0.8, zorder=1)
    ax.text(d * 0.97, d * 0.97, "σ = effect", color=MUTED, fontsize=7, ha="right", va="top")
    ax.scatter(mu, sg, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    # Label the most influential points; push labels apart vertically so they do not overlap.
    top = [i for i in np.argsort(-mu)[:7]]
    ys = {i: sg[i] for i in top}
    gap = ymax * 0.075
    for _ in range(50):
        moved = False
        for a in top:
            for b in top:
                if a != b and abs(mu[a] - mu[b]) < xmax * 0.28 and abs(ys[a] - ys[b]) < gap:
                    lo, hi = (a, b) if ys[a] <= ys[b] else (b, a)
                    ys[lo] -= gap * 0.25
                    ys[hi] += gap * 0.25
                    moved = True
        if not moved:
            break
    for i in top:
        ax.annotate(label(names[i]), (mu[i], sg[i]), xytext=(mu[i] + xmax * 0.015, ys[i]), textcoords="data",
                    fontsize=7.2, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_xlabel("effect on the score (mean |change|)")
    ax.set_ylabel("σ of the change")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_title("Effect against interaction: high σ means the effect depends on the other parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reps = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in RUNS.items() if p.exists()}
    figs = {}
    for m, rep in reps.items():
        f = {}
        f["score"] = FIG_DIR / f"{m}-score.png"
        fig_effect(rep, "score", "Effect of each parameter on the matching score",
                   "mean |change| of the score for a step of 2/3 of the parameter's range", f["score"])
        f["direction"] = FIG_DIR / f"{m}-direction.png"
        fig_direction(rep, f["direction"])
        f["parts"] = FIG_DIR / f"{m}-parts.png"
        fig_parts(rep, f["parts"])
        f["plane"] = FIG_DIR / f"{m}-plane.png"
        fig_plane(rep, f["plane"])
        f["contacts"] = FIG_DIR / f"{m}-contacts.png"
        fig_effect(rep, "contact_share", "Effect of each parameter on touching pairs per vehicle",
                   "mean |change| of touching pairs per vehicle for the same step", f["contacts"])
        figs[m] = f
    return reps, figs


# ------------------------------------------------------------------ the PDF

def build_pdf(reps, figs, text):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph, SimpleDocTemplate, Table,
                                    TableStyle)
    F = "C:/Windows/Fonts/"
    pdfmetrics.registerFont(TTFont("Body", F + "segoeui.ttf"))
    pdfmetrics.registerFont(TTFont("Semi", F + "seguisb.ttf"))
    pdfmetrics.registerFont(TTFont("Mono", F + "consola.ttf"))
    registerFontFamily("Body", normal="Body", bold="Semi", italic="Body", boldItalic="Semi")
    ink, muted, accent, rule, shade = (colors.HexColor(c) for c in
                                       ("#1f2328", "#59636e", "#2f5f8f", "#d0d7de", "#f3f5f7"))
    body = ParagraphStyle("b", fontName="Body", fontSize=9.6, leading=13.6, textColor=ink, spaceAfter=6)
    cell = ParagraphStyle("c", parent=body, fontSize=8.2, leading=10.8, spaceAfter=0)
    cellb = ParagraphStyle("cb", parent=cell, fontName="Semi")
    cap = ParagraphStyle("cap", parent=body, fontSize=8.2, leading=11, textColor=muted, spaceBefore=2, spaceAfter=10)
    h1 = ParagraphStyle("h1", fontName="Semi", fontSize=15, leading=19, textColor=accent, spaceBefore=6, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName="Semi", fontSize=11.2, leading=15, textColor=ink, spaceBefore=8, spaceAfter=4)
    title = ParagraphStyle("t", fontName="Semi", fontSize=22, leading=27, textColor=ink, spaceAfter=4)
    sub = ParagraphStyle("s", parent=body, fontSize=10.5, leading=15, textColor=muted, spaceAfter=12)
    bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
    W = A4[0] - 32 * mm

    def P(t, st=body):
        return Paragraph(t, st)

    def B(items):
        return [Paragraph(t, bullet, bulletText="•") for t in items]

    def img(path, width=W):
        from PIL import Image as PI
        w, h = PI.open(path).size
        return Image(str(path), width=width, height=width * h / w)

    def table(rows, widths):
        data = [[Paragraph(str(c), cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, rule),
                               ("BACKGROUND", (0, 0), (-1, 0), shade), ("LINEBELOW", (0, 0), (-1, 0), 0.8, muted),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        return t

    def ranking(rep):
        e, noise = rep["effects"], rep["noise_floor"]["score"]
        rows = [["Parameter", "Effect on score", "× noise", "Parts it moves most", "Touching pairs per vehicle"]]
        for n in sorted(rep["names"], key=lambda n: -e[n]["score"]["effect"]):
            parts = sorted(((k[5:], v["effect"]) for k, v in e[n].items() if k.startswith("part:")),
                           key=lambda x: -x[1])
            c = e[n]["contact_share"]
            rows.append([label(n), f"{e[n]['score']['effect']:.3f}", f"{e[n]['score']['effect'] / noise:.1f}",
                         ", ".join(p.replace("_", " ") for p, _ in parts[:2]),
                         f"{c['effect']:.2f} ({'up' if c['mean'] >= 0 else 'down'} when raised)"])
        return rows

    story = [P("Parameter sensitivity of the matching score", title), P(text["subtitle"], sub)]
    story += [P("Summary", h1)] + B(text["summary"])
    story += [P("1. Method", h1)] + [P(t) for t in text["method_intro"]] + B(text["method"])
    for k, (m, name) in enumerate((("choice", "Choice model"), ("priority", "Priority model")), 2):
        rep, f = reps[m], figs[m]
        story += [PageBreak(), P(f"{k}. {name}", h1)] + [P(t) for t in text[m]]
        story += [img(f["score"]), P("Effect on the score; the longest bar is the most influential parameter. "
                                     "Dashed line: seed noise; dotted: twice the noise.", cap)]
        story += [CondPageBreak(240), img(f["direction"]),
                  P("Signed mean change when the parameter is raised (blue: the simulation gets closer to the video "
                    "on average; red: further away), with ± σ. A line that crosses zero means the direction depends "
                    "on the other parameters.", cap)]
        story += [CondPageBreak(280), img(f["parts"]),
                  P("Effect on each part of the score: <i>how</i> a parameter changes the match.", cap)]
        story += [CondPageBreak(230), img(f["plane"], W * 0.92),
                  P("Morris plane. Further right: more influential. Near or above the diagonal: the effect is mostly "
                    "interaction or non-linear.", cap)]
        story += [CondPageBreak(240), img(f["contacts"]),
                  P("Effect on touching pairs per vehicle, the safety side of the fitting objective.", cap)]
        story += [CondPageBreak(160), P("Ranking", h2),
                  table(ranking(rep), [W * 0.21, W * 0.12, W * 0.08, W * 0.29, W * 0.30])]
    story += [PageBreak(), P("4. What this means for the optimisation", h1)] + B(text["implications"])
    story += [P("Caveats", h2)] + B(text["caveats"])

    def on_page(c, d):
        c.saveState()
        c.setFont("Body", 7.5)
        c.setFillColor(muted)
        c.drawString(16 * mm, 10 * mm, "Parameter sensitivity of the matching score")
        c.drawRightString(A4[0] - 16 * mm, 10 * mm, str(d.page))
        c.restoreState()

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="Parameter sensitivity of the matching score")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    reps, figs = make_figures()
    text = json.loads((Path(__file__).with_name("sensitivity_text.json")).read_text(encoding="utf-8"))
    build_pdf(reps, figs, text)
    print("wrote", OUT_PDF)
    sys.exit(0)
