"""Builds the figures and tables in docs/thesis-prep from the run outputs in runs/.

    .venv\\Scripts\\python.exe docs\\thesis-prep\\build.py            # figures in English and Vietnamese, tables
    .venv\\Scripts\\python.exe docs\\thesis-prep\\build.py --lang vi  # one language

Writes figures/<lang>/*.pdf (vector, for \\includegraphics) and *.png (preview), tables/*.tex (LaTeX
tabular, paste into a table environment) and tables/*.csv (the same numbers). Every number comes from a
file under runs/ or videos/.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "twin"
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ style (dataviz reference palette)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
# Categorical slots 1-3, validated (validate_palette.js, light): colour follows the model everywhere.
MODEL_COLOR = {"choice": "#2a78d6", "priority": "#eb6834", "constant": "#1baf7a"}
SEQ_BLUE = "#2a78d6"
NEUTRAL = "#c9c8c2"

for f in ("segoeui.ttf", "seguisb.ttf"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.titlelocation": "left", "axes.titlepad": 8, "legend.frameon": False, "pdf.fonttype": 42,
})

# ------------------------------------------------------------------ labels in both languages

T = {
    "en": {
        "constant": "Constant speed", "choice": "Choice model", "priority": "Priority model",
        "choice_default": "Choice, defaults", "choice_fitted": "Choice, fitted",
        "priority_start": "Priority, start", "priority_fitted": "Priority, fitted",
        "fit": "Fit clips (6 Nov)", "holdout": "Held-out clips (8 Nov)",
        "score": "Image-space score (lower is closer)", "contacts": "Touching pairs per vehicle",
        "generation": "Generation", "objective": "Objective", "best_so_far": "Best so far",
        "median": "Median of generation",
        "conv_choice": "Choice model: 6 parameters, 25 generations of 12",
        "conv_priority": "Priority model: 25 parameters, 80 × 20, contact weight 3",
        "parts_title": "Score by part, held-out clips",
        "parts": {"occupancy_map": "Occupancy map", "occupancy_time": "Occupancy over time", "motion": "Motion",
                  "spacing": "Spacing", "crossings": "Line crossings", "headways": "Headways"},
        "recovery_x": "Error of the fitted value, share of the parameter's range",
        "recovery_title": "Recovery test: fitting a simulation with known parameters",
        "found": "found (within 10%)", "not_found": "not found",
        "scan_y": "Objective (mean of 2 seeds, 12 clips)", "scan_title": "One parameter at a time, others at defaults",
        "default": "default", "noise": "noise band",
        "noise_x": "Distance from the camera (m)", "noise_y": "Position noise, SD (m)",
        "noise_title": "Tracking noise by distance from the camera (12 clips)", "clip_range": "range over clips",
        "mean": "mean",
        "ct_x": "Simulated time (s)", "ct_y": "Pairs of vehicles that have touched",
        "ct_title": "Contacts accumulating over one replayed clip (Bưởi, 6 Nov, segment 0010)",
        "score_title": "Score by model", "contacts_title": "Contacts by model",
    },
    "vi": {
        "constant": "Tốc độ không đổi", "choice": "Mô hình lựa chọn", "priority": "Mô hình ưu tiên",
        "choice_default": "Lựa chọn, mặc định", "choice_fitted": "Lựa chọn, đã hiệu chỉnh",
        "priority_start": "Ưu tiên, ban đầu", "priority_fitted": "Ưu tiên, đã hiệu chỉnh",
        "fit": "Clip hiệu chỉnh (6/11)", "holdout": "Clip kiểm tra (8/11)",
        "score": "Điểm sai khác trên ảnh (càng thấp càng gần)", "contacts": "Số cặp va chạm trên mỗi xe",
        "generation": "Thế hệ", "objective": "Hàm mục tiêu", "best_so_far": "Tốt nhất đến hiện tại",
        "median": "Trung vị của thế hệ",
        "conv_choice": "Mô hình lựa chọn: 6 tham số, 25 thế hệ × 12",
        "conv_priority": "Mô hình ưu tiên: 25 tham số, 80 × 20, trọng số va chạm 3",
        "parts_title": "Điểm theo từng thành phần, clip kiểm tra",
        "parts": {"occupancy_map": "Bản đồ chiếm chỗ", "occupancy_time": "Chiếm chỗ theo thời gian",
                  "motion": "Chuyển động", "spacing": "Khoảng cách", "crossings": "Lượt qua vạch",
                  "headways": "Giãn cách thời gian"},
        "recovery_x": "Sai số giá trị hiệu chỉnh, tỉ lệ so với miền tham số",
        "recovery_title": "Kiểm tra khôi phục: hiệu chỉnh trên mô phỏng có tham số đã biết",
        "found": "khôi phục được (sai số < 10%)", "not_found": "không khôi phục được",
        "scan_y": "Hàm mục tiêu (trung bình 2 seed, 12 clip)", "scan_title": "Quét từng tham số, các tham số khác giữ mặc định",
        "default": "mặc định", "noise": "dải nhiễu",
        "noise_x": "Khoảng cách tới camera (m)", "noise_y": "Nhiễu vị trí, độ lệch chuẩn (m)",
        "noise_title": "Nhiễu bám vết theo khoảng cách tới camera (12 clip)", "clip_range": "khoảng giữa các clip",
        "mean": "trung bình",
        "ct_x": "Thời gian mô phỏng (s)", "ct_y": "Số cặp xe đã va chạm",
        "ct_title": "Va chạm tích luỹ trong một clip tái hiện (Bưởi, 6/11, đoạn 0010)",
        "score_title": "Điểm theo mô hình", "contacts_title": "Va chạm theo mô hình",
    },
}
PARTS = ["motion", "spacing", "occupancy_map", "occupancy_time", "crossings", "headways"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig, name, lang):
    out = HERE / "figures" / lang
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def hbars(ax, y, w, color, height=0.62):
    bars = ax.barh(y, w, height=height, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    return bars


def recessive_x(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def recessive_y(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


# ------------------------------------------------------------------ the numbers

CONTACT_WEIGHT, JAM_WEIGHT = 0.5, 1.0   # the objective every row is compared under (twin/calibrate/search.py)


def objective(entry):
    """The objective at contact weight 0.5, whatever weight the run itself was searched with."""
    return entry["score"] + CONTACT_WEIGHT * entry["contact_share"] + JAM_WEIGHT * entry["jammed"]


EVAL = RUNS / "results" / "report.json"
CONFIGS = (("constant", "constant"), ("choice_default", "choice"), ("choice_fitted", "choice"),
           ("priority_start", "priority"), ("priority_fitted", "priority"))
# Figures and tables made from the evaluation; removed while it has not run, so no stale numbers remain.
RESULT_FIGURES = ("score-by-model", "contacts-by-model", "score-parts")
RESULT_TABLES = ("results-main", "results-parts-holdout")


def results():
    """The rows of the main comparison, (key, model colour, entry with "fit" and "holdout"), from the
    evaluation (python -m twin.calibrate.evaluate, labels as in CONFIGS). None until it has run."""
    if not EVAL.exists():
        return None
    configs = load(EVAL)["configs"]
    rows = [(key, model, configs[key]) for key, model in CONFIGS
            if configs.get(key, {}).get("fit") and configs.get(key, {}).get("holdout")]
    return rows if len(rows) == len(CONFIGS) else None


# ------------------------------------------------------------------ figures

def fig_scores(lang):
    t, rows = T[lang], results()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, side in zip(axes, ("fit", "holdout")):
        y = np.arange(len(rows))[::-1]
        vals = [r[2][side]["score"] for r in rows]
        hbars(ax, y, vals, [MODEL_COLOR[r[1]] for r in rows])
        for yy, v in zip(y, vals):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", color=INK2, fontsize=8)
        ax.set_yticks(y, [t[r[0]] for r in rows])
        ax.set_xlim(0, 0.4)
        ax.set_title(t[side])
        recessive_x(ax)
    axes[0].set_xlabel(t["score"])
    axes[1].set_xlabel(t["score"])
    save(fig, "score-by-model", lang)


def fig_contacts(lang):
    t, rows = T[lang], results()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, side in zip(axes, ("fit", "holdout")):
        y = np.arange(len(rows))[::-1]
        vals = [r[2][side]["contact_share"] for r in rows]
        hbars(ax, y, vals, [MODEL_COLOR[r[1]] for r in rows])
        for yy, v in zip(y, vals):
            ax.text(v + 0.02, yy, f"{v:.2f}", va="center", color=INK2, fontsize=8)
        ax.set_yticks(y, [t[r[0]] for r in rows])
        ax.set_xlim(0, 1.55)
        ax.set_title(t[side])
        ax.set_xlabel(t["contacts"])
        recessive_x(ax)
    save(fig, "contacts-by-model", lang)


def fig_parts(lang):
    t = T[lang]
    rows = [r for r in results() if r[0] in ("constant", "choice_fitted", "priority_fitted")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n, h = len(rows), 0.26
    base = np.arange(len(PARTS))[::-1]
    for k, (key, model, rep) in enumerate(rows):
        y = base + (n - 1 - 2 * k) * h / 2
        vals = [rep["holdout"]["parts"].get(p, np.nan) for p in PARTS]
        hbars(ax, y, vals, MODEL_COLOR[model], height=h * 0.92)
        for yy, v in zip(y, vals):
            ax.text(v + 0.005, yy, f"{v:.2f}", va="center", color=INK2, fontsize=7)
    ax.set_yticks(base, [t["parts"][p] for p in PARTS])
    ax.set_xlim(0, 0.5)
    ax.set_xlabel(t["score"])
    ax.set_title(t["parts_title"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for _, m, _ in rows]
    ax.legend(handles, [t[k] for k, _, _ in rows], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=8)
    recessive_x(ax)
    save(fig, "score-parts", lang)


def fig_convergence(lang):
    t = T[lang]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, run, model, title in ((axes[0], "fit2", "choice", t["conv_choice"]),
                                  (axes[1], "priority-sep", "priority", t["conv_priority"])):
        h = load(RUNS / run / "report.json")["history"]
        g = [r["generation"] for r in h]
        ax.plot(g, [r["median"] for r in h], color=NEUTRAL, linewidth=1.5, label=t["median"])
        ax.plot(g, [r["best_so_far"] for r in h], color=MODEL_COLOR[model], linewidth=2, label=t["best_so_far"],
                solid_capstyle="round")
        last = h[-1]["best_so_far"]
        ax.annotate(f"{last:.3f}", (g[-1], last), xytext=(0, 6), textcoords="offset points", color=INK2, fontsize=8,
                    ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(t["generation"])
        recessive_y(ax)
    axes[0].set_ylabel(t["objective"])
    axes[0].legend(fontsize=8, loc="upper right")
    save(fig, "convergence", lang)


def fig_recovery(lang):
    t = T[lang]
    params = load(RUNS / "recovery/report.json")["parameters"]
    names = sorted(params, key=lambda n: params[n]["error_share_of_range"])[::-1]
    err = [params[n]["error_share_of_range"] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(names))
    hbars(ax, y, err, [SEQ_BLUE if e < 0.1 else NEUTRAL for e in err])
    for yy, e in zip(y, err):
        ax.text(e + 0.006, yy, f"{e:.0%}", va="center", color=INK2, fontsize=8)
    ax.axvline(0.1, color=MUTED, linewidth=1, zorder=1)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 0.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(t["recovery_x"])
    ax.set_title(t["recovery_title"], fontsize=9)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=SEQ_BLUE), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)],
              [t["found"], t["not_found"]], loc="upper right", fontsize=8)
    recessive_x(ax)
    save(fig, "recovery", lang)


def fig_scan(lang):
    """One-at-a-time scan of the choice model over 12 clips (runs/twin/scan.json)."""
    t = T[lang]
    from twin.behaviour.params import BY_NAME
    scan = load(RUNS / "scan.json")
    names = list(scan)
    cols = 5
    rows_n = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.2, 1.75 * rows_n), sharey=True)
    floor = None
    for ax, name in zip(axes.flat, names):
        xs = sorted(scan[name], key=float)
        x = np.array([float(v) for v in xs])
        obj = np.array([scan[name][v]["objective"] for v in xs])
        seeds = np.array([[scan[name][v]["d6"], scan[name][v]["d8"]] for v in xs])
        ax.fill_between(x, seeds.min(1), seeds.max(1), color=SEQ_BLUE, alpha=0.12, linewidth=0)
        ax.plot(x, obj, color=SEQ_BLUE, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE,
                markeredgewidth=1)
        d = BY_NAME[name].default if name in BY_NAME else None
        if d is not None:
            ax.axvline(d, color=MUTED, linewidth=0.8, zorder=0)
        if BY_NAME.get(name) and BY_NAME[name].log:
            ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(x, [f"{v:g}" for v in x])     # the scanned values themselves
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        recessive_y(ax)
        floor = obj.min() if floor is None else min(floor, obj.min())
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(t["objective"], fontsize=8)
    fig.suptitle(t["scan_title"], x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, "scan-one-at-a-time", lang)


def fig_tracking_noise(lang):
    t = T[lang]
    rows = load(RUNS / "tracking-error.json")
    bands = list(rows[0]["position_noise_by_camera_distance_sd_m"])
    x = np.array([int(b.split("-")[0]) + 5 for b in bands])
    vals = np.array([[r["position_noise_by_camera_distance_sd_m"].get(b, np.nan) for b in bands] for r in rows], float)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.fill_between(x, np.nanmin(vals, 0), np.nanmax(vals, 0), color=SEQ_BLUE, alpha=0.12, linewidth=0,
                    label=t["clip_range"])
    m = np.nanmean(vals, 0)
    ax.plot(x, m, color=SEQ_BLUE, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE, label=t["mean"])
    ax.annotate(f"{m[-1]:.2f} m", (x[-1], m[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel(t["noise_x"])
    ax.set_ylabel(t["noise_y"])
    ax.set_title(t["noise_title"], fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper left")
    recessive_y(ax)
    save(fig, "tracking-noise", lang)


def fig_contact_timeline(lang):
    t = T[lang]
    path = ROOT / "videos/three-models/buoi-hht/contact_times.json"
    if not path.exists():
        return
    data = load(path)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ends = {}
    for model in ("constant", "choice", "priority"):
        times = sorted(data.get(model, []))
        if times:
            ax.step([0] + times, np.arange(len(times) + 1), where="post", color=MODEL_COLOR[model], linewidth=2)
            ends[model] = (times[-1], len(times))
    # End labels, nudged apart when two lines end close together.
    order = sorted(ends, key=lambda m: ends[m][1])
    for k, model in enumerate(order):
        x_end, n = ends[model]
        nudge = 0 if k == 0 or n - ends[order[k - 1]][1] > 15 else 9
        ax.annotate(f"{t[model]}: {n}", (x_end, n), xytext=(4, nudge - (6 if k == 0 and len(order) > 1 and
                    ends[order[1]][1] - n <= 15 else 0)), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    ax.set_xlabel(t["ct_x"])
    ax.set_ylabel(t["ct_y"])
    ax.set_title(t["ct_title"], fontsize=9)
    ax.set_xlim(0, None)
    recessive_y(ax)
    save(fig, "contacts-over-time", lang)


def copy_existing(lang):
    """Figures already made by other tools (English labels only)."""
    out = HERE / "figures" / lang
    for src, name in ((RUNS / "sensitivity-figures/choice-score.png", "sensitivity-choice-score.png"),
                      (RUNS / "sensitivity-figures/priority-score.png", "sensitivity-priority-score.png"),
                      (RUNS / "sensitivity-figures/choice-parts.png", "sensitivity-choice-parts.png"),
                      (RUNS / "sensitivity-figures/choice-plane.png", "sensitivity-choice-plane.png"),
                      (RUNS / "average-speed.png", "average-speed.png")):
        if src.exists():
            shutil.copyfile(src, out / name)


# ------------------------------------------------------------------ tables

def write_table(name, header, rows, *, align=None, caption_hint="", tex_header=None):
    """tex_header: LaTeX header lines to use instead of one bold row of `header` (the CSV keeps `header`)."""
    out = HERE / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    def tex(x):
        s = str(x)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                     ("~", r"\textasciitilde{}"), ("→", r"$\rightarrow$"), ("≤", r"$\le$"), ("×", r"$\times$")):
            s = s.replace(a, b)
        return s

    align = align or ("l" + "r" * (len(header) - 1))
    lines = [f"% {caption_hint}" if caption_hint else "% generated by docs/thesis-prep/build.py",
             f"\\begin{{tabular}}{{|{'|'.join(align)}|}}", r"\hline",
             *(tex_header or [" & ".join(rf"\textbf{{{tex(h)}}}" for h in header) + r" \\"]), r"\hline"]
    lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}", ""]
    (out / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def f3(x):
    return "" if x is None else f"{x:.3f}"


def tables():
    from twin.behaviour import default_params
    from twin.behaviour.params import SPECS

    # Dataset and tracking quality
    rows = []
    for r in load(RUNS / "tracking-error.json"):
        name = Path(r["file"]).name.replace(".trajectories.json", "")
        day = "6 Nov" if "20261106" in name else "8 Nov"
        seg, clock = name.split("seg-")[1].split("-")
        rows.append([f"{day} {clock[:2]}:{clock[2:4]}", seg, "fit" if day == "6 Nov" else "held out", r["raw_tracks"],
                     r["stitched_tracks"], r["line_crossings"], f"{r['position_noise_sd_m']:.2f}",
                     f"{r['off_layout_share']:.0%}"])
    write_table("dataset", ["Start", "Segment", "Use", "Raw tracks", "Stitched", "Crossings",
                            "Noise (m)", "Off layout"], rows, align="lllrrrrr",
                caption_hint="Bưởi - Hoàng Hoa Thám, 12 clips of 3 minutes, 384x216 at 25 fps")

    # Parameters
    write_table("params-choice", ["Parameter", "Default", "Range", "Scale", "Meaning"],
                [[s.name, f"{s.default:g}", f"{s.lo:g} - {s.hi:g}", "log" if s.log else "linear", s.doc] for s in SPECS],
                align=["l", "r", "r", "l", "p{6.5cm}"])
    pp = default_params("priority").specs
    pp_rows = [[s.name, f"{s.default:g}", f"{s.lo:g} - {s.hi:g}", "log" if s.log else "linear", s.doc] for s in pp]
    pp_rows.insert(1, ["v_r<r>_k<k>", "7", "1.5 - 13", "log",
                       "speed on busy route r at point k of 4 along it (m/s); 16 at Bưởi"])
    pp_rows.append(["turn_<k>", "0", "-2.5 - 2.5", "linear",
                    "demand, not behaviour: log-odds shift of the main entry's route k (2 at Bưởi)"])
    write_table("params-priority", ["Parameter", "Default", "Range", "Scale", "Meaning"], pp_rows,
                align=["l", "r", "r", "l", "p{6.5cm}"])

    # Score parts
    from twin.measure.image_score import DEFAULT_WEIGHTS
    meaning = {"occupancy_map": "where vehicles are over the clip: 16 px cells, blurred; Bray-Curtis",
               "occupancy_time": "vehicles in each of 6 x 4 zones per 10 s; Bray-Curtis",
               "motion": "image velocity over 0.6 s / road scale, per zone and group; W1 / mean speed",
               "spacing": "distance to the nearest vehicle / road scale, per group; W1 / mean",
               "crossings": "cumulative crossings of each counting line; area between curves / area under upper",
               "headways": "time between successive crossings of a line; W1 / mean"}
    write_table("score-parts", ["Part", "Weight", "Compares"],
                [[p, f"{DEFAULT_WEIGHTS[p]:g}", meaning[p]] for p in PARTS], align=["l", "r", "p{9cm}"])

    # Main results
    res = results()
    if res is not None:
        rows = [[T["en"][key], f3(rep["fit"]["score"]), f3(objective(rep["fit"])), f"{rep['fit']['contact_share']:.2f}",
                 f3(rep["holdout"]["score"]), f3(objective(rep["holdout"])), f"{rep['holdout']['contact_share']:.2f}"]
                for key, _, rep in res]
        write_table("results-main", ["Model", "Fit score", "Fit objective", "Fit contacts", "Held-out score",
                                     "Held-out objective", "Held-out contacts"], rows,
                    tex_header=[r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{Fit clips (6 Nov)}} & "
                                r"\multicolumn{3}{c|}{\textbf{Held-out clips (8 Nov)}} \\", r"\cline{2-7}",
                                r" & Score & Objective & Contacts & Score & Objective & Contacts \\"],
                    caption_hint="Projected simulation vs tracked video; means over 6 clips x 3 seeds; "
                                 "objective = score + 0.5 x contacts per vehicle + jammed share")
        write_table("results-parts-holdout", ["Model"] + PARTS,
                    [[T["en"][key]] + [f3(rep["holdout"]["parts"].get(p)) for p in PARTS] for key, _, rep in res])

    # Fitted parameters
    for run, name in (("fit2", "fitted-choice"), ("priority-sep", "fitted-priority")):
        p = load(RUNS / run / "report.json")["parameters"]
        write_table(name, ["Parameter", "Start", "Fitted", "Range"],
                    [[n, f"{r['start']:g}", f"{r['fitted']:.3g}", f"{r['bounds'][0]:g} - {r['bounds'][1]:g}"]
                     for n, r in p.items()])

    # Recovery
    p = load(RUNS / "recovery/report.json")["parameters"]
    write_table("recovery", ["Parameter", "Truth", "Start", "Fitted", "Error (share of range)", "Found"],
                [[n, f"{r['truth']:g}", f"{r['start']:g}", f"{r['fitted']:.3g}", f"{r['error_share_of_range']:.0%}",
                  "yes" if r["error_share_of_range"] < 0.1 else "no"] for n, r in p.items()],
                caption_hint="Choice model, 3 simulated clips, 30 generations; run before the keep-clear cost was added")

    # Sensitivity ranking
    for model in ("choice", "priority"):
        rep = load(RUNS / f"sensitivity-{model}/sensitivity.json")
        eff, noise = rep["effects"], rep["noise_floor"]["score"]
        names = sorted(rep["names"], key=lambda n: -eff[n]["score"]["effect"])
        write_table(f"sensitivity-{model}", ["Parameter", "mu* (effect)", "sigma", "mean (signed)", "above noise"],
                    [[n, f"{eff[n]['score']['effect']:.4f}", f"{eff[n]['score']['sigma']:.4f}",
                      f"{eff[n]['score']['mean']:+.4f}", "yes" if eff[n]["score"]["effect"] > noise else "no"]
                     for n in names],
                    caption_hint=f"Morris, {rep['trajectories']} trajectories x {len(rep['clips'])} clips; "
                                 f"noise floor {noise:.4f}; {rep['wall_s'] / 60:.0f} min")

    # Compute cost
    rows = []
    for run, label in (("fit2", "Choice fit (6 params)"), ("priority", "Priority fit (25 params)"),
                       ("priority-cw3", "Priority fit, contact weight 3"), ("priority-sep", "Priority + separating, weight 3"),
                       ("recovery", "Recovery test (12 params)")):
        rep = load(RUNS / run / "report.json")
        g = len(rep["history"])
        rows.append([label, g, len(rep["fit_clips"]), len(rep["holdout_clips"]), f"{rep['wall_s'] / 60:.1f}"])
    for model in ("choice", "priority"):
        rep = load(RUNS / f"sensitivity-{model}/sensitivity.json")
        rows.append([f"Sensitivity, {model} ({len(rep['names'])} params)", "-", len(rep["clips"]), 0,
                     f"{rep['wall_s'] / 60:.1f}"])
    write_table("compute-cost", ["Run", "Generations", "Fit clips", "Held-out clips", "Wall time (min)"], rows,
                caption_hint="One desktop machine, all cores; 3-minute clips")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "vi", "both"), default="both")
    args = ap.parse_args(argv)
    have_results = results() is not None
    figures = [fig_convergence, fig_recovery, fig_scan, fig_tracking_noise, fig_contact_timeline, copy_existing]
    if have_results:
        figures = [fig_scores, fig_contacts, fig_parts] + figures
    for lang in (("en", "vi") if args.lang == "both" else (args.lang,)):
        for f in figures:
            f(lang)
        if not have_results:
            for name in RESULT_FIGURES:
                for ext in ("pdf", "png"):
                    (HERE / "figures" / lang / f"{name}.{ext}").unlink(missing_ok=True)
    if not have_results:
        for name in RESULT_TABLES:
            for ext in ("tex", "csv"):
                (HERE / "tables" / f"{name}.{ext}").unlink(missing_ok=True)
        print(f"No evaluation at {EVAL.relative_to(ROOT)}: the result figures and tables "
              f"({', '.join(RESULT_FIGURES + RESULT_TABLES)}) are left out; run python -m twin.calibrate.evaluate "
              f"(docs/twin.md section 5b).")
    tables()
    print(f"figures -> {HERE / 'figures'}\ntables  -> {HERE / 'tables'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
