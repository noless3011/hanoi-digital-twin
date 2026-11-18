# Thesis preparation material

English source material for writing the thesis (in Vietnamese) in `thesis/`: text to work from,
algorithms, figures and tables. Every number comes from the run outputs in `runs/twin/` and
`videos/`; `build.py` regenerates the figures and tables from them.

**Method decision:** the simulation is compared with the tracked video by projecting each simulated
vehicle's box into the camera image, not by rendering it and passing it through RF-DETR + TrackTrack.
The detector path takes about 6 minutes per clip on one GPU (17 hours for the evaluation, impossible
inside a search); projection takes 0.1 s. What this costs is in [`projection.md`](projection.md).

**Status:** the evaluation has run (`runs/twin/results/report.json`, 17 Nov 2026, about 3 minutes;
command in `docs/twin.md` section 5b), so every figure and table below exists.

## Where things go in the thesis

| Thesis chapter (`thesis/Chuong/`) | Material |
| --- | --- |
| 3 Đề xuất (method) | [`01-method.md`](01-method.md) (with a revised section outline), [`algorithms.tex`](algorithms.tex), `figures/architecture.pdf`; [`projection.md`](projection.md) for why the simulation is projected (also 5.6 and 6) |
| 4 Triển khai (implementation) | [`02-implementation.md`](02-implementation.md) |
| 5 Thực nghiệm (experiments) | [`03-experiments.md`](03-experiments.md), organised by the chapter's existing sections; all figures and tables |
| 6 Kết luận, abstract | [`04-conclusion.md`](04-conclusion.md) |
| Tài liệu tham khảo | [`references.bib`](references.bib): candidates not yet in the thesis bibliography (check each before citing) |

## Figures

`figures/vi/` has Vietnamese labels, `figures/en/` English. Use the `.pdf` (vector) in the thesis:
`\includegraphics[width=\textwidth]{../docs/thesis-prep/figures/vi/score-by-model.pdf}`, or copy the file
into `thesis/Hinh_ve/`.

| File | Shows | Section |
| --- | --- | --- |
| `figures/architecture.pdf` | The system and the calibration loop (TikZ source `architecture.tex`, English labels; edit it to translate) | 3.1 |
| `score-by-model` | Image-space score of the five model configurations, fit and held-out clips | 5.4 |
| `contacts-by-model` | Touching pairs per vehicle, same configurations | 5.4 |
| `score-parts` | The six score parts on held-out clips: constant speed vs fitted choice vs fitted priority | 5.4 |
| `convergence` | CMA-ES best-so-far and median objective per generation, both models | 5.4 |
| `recovery` | Recovery test: error of each recovered parameter, 10% threshold | 5.3 |
| `scan-one-at-a-time` | Objective along each choice-model parameter, others at defaults; band = two seeds | 5.6 |
| `tracking-noise` | Position noise by distance from the camera, mean and range over 12 clips | 5.2 |
| `contacts-over-time` | Touching pairs accumulating over one replayed clip, three models | 5.7 |
| `sensitivity-*.png`, `average-speed.png` | Copied from earlier reports (English only, PNG) | 5.6, 5.2 |

Colours follow the model everywhere: constant speed green-aqua, choice blue, priority orange.

## Tables

`tables/<name>.tex` is a `tabular` for `\input{}` inside your own `table` environment (add the caption
and label in Vietnamese; the first line of each file is a comment with what the caption must say).
`tables/<name>.csv` has the same numbers. All fit the thesis text width (checked with the thesis
margins at `\small`).

| Table | Content | Section |
| --- | --- | --- |
| `dataset` | The 12 clips: date, segment, fit/held-out, tracks, crossings, noise, off-layout share | 5.2 |
| `params-choice`, `params-priority` | Every parameter: default, range, scale, meaning | 3.4 |
| `score-parts` | The six parts of the score, their weights and distances | 3.6 |
| `results-main` | Score, objective and contacts per model, fit and held-out | 5.4 |
| `results-parts-holdout` | The six parts per model, held-out clips | 5.4 |
| `fitted-choice`, `fitted-priority` | Start and fitted values | 5.4 |
| `recovery` | Truth, start, fitted, error per parameter | 5.3 |
| `sensitivity-choice`, `sensitivity-priority` | Morris $\mu^\ast$, $\sigma$, signed mean, above the noise floor | 5.6 |
| `compute-cost` | Generations, clips, wall time per run | 5.6 |

## Algorithms

[`algorithms.tex`](algorithms.tex): seven `algorithm2e` blocks (the package `DoAn.tex` already loads):
simulation step, choice model, priority model, image-space score, CMA-ES calibration, Morris
sensitivity, demand extraction. They compile with the thesis preamble; translate captions and comments.

## Things to know before writing

1. **The priority model in the results is the final one** (`runs/twin/priority-sep`, with the
   separating rules). `docs/twin.md` also reports `runs/twin/priority`, an earlier version without them.
2. **Objectives use contact weight 0.5 for every row**, so configurations searched at weight 3 can be
   compared.
3. **The recovery test predates the "keep clear" cost** in the choice model. Re-run it if section 5.3
   must describe the current model exactly (about 25 minutes).
4. **Missing for chapter 5**: an optimiser comparison (5.5) and repeated fits for stability (5.6).
   `03-experiments.md` says what to run for each.
5. Section 3's current outline in the thesis describes the old C# model; `01-method.md` proposes an
   outline matching what was built.

## Rebuild

```powershell
.\.venv\Scripts\python.exe docs\thesis-prep\build.py           # both languages, all tables
.\.venv\Scripts\python.exe docs\thesis-prep\build.py --lang vi
cd docs\thesis-prep\figures; pdflatex architecture.tex           # the architecture diagram
```

After a new fit or sensitivity run, rebuild and the figures and tables follow.
