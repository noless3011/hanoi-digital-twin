# Fine-tuning data: Hanoi frames and other Vietnamese datasets

Tools to build detector fine-tuning data in the seven-class COCO layout of [`../training`](../training/README.md).
Standard library plus OpenCV/Pillow only; the labeller is a single local page.

| File | Purpose |
| --- | --- |
| `sample_frames.py` | Picks diverse frames from the recorded videos in `data/` and pre-draws boxes with the current RF-DETR |
| `label_server.py`, `label_ui.html` | Local labeller at http://localhost:8000 |
| `export_coco.py` | `hanoi`: labelled frames → train/valid/test (split by video). `yolo`: YOLO-format dataset (e.g. Kaggle vehicle-danang-2025) → same layout |

## 1. Sample frames

```powershell
.\.venv\Scripts\python.exe perception\finetune\sample_frames.py --project data\labeling\hanoi-v1 --per-camera 100
```

Frames are spread over every recording of each camera (camera = data folder name before its date), at
least 15 s apart within a video, near-duplicates skipped. Running it again with a larger `--per-camera`
adds frames to the same project.

## 2. Label

Double-click `label.bat` in the project root (or run `.\label.bat [project folder]`). It samples frames
first if the project does not exist yet, starts the server and opens http://localhost:8000; Ctrl+C in its
window stops it. The same by hand:

```powershell
.\.venv\Scripts\python.exe perception\finetune\label_server.py --project data\labeling\hanoi-v1
```
 Each frame starts with the current model's boxes ("pre-labelled"): fix the
class of wrong ones, delete false ones, draw the missed ones, then press **Enter** (done, next frame).

| Action | How |
| --- | --- |
| New box | drag on empty space (current class) |
| Select / move / resize | click a box / drag inside / drag a corner or edge |
| Class | keys **1–6** (car, bus, van, others, motorcycle, ignore); changes the selected box too |
| Delete, undo | **Del**, **Ctrl+Z** |
| Zoom, pan, fit | mouse wheel, right-drag or Space+drag, **F** |
| Hide boxes | **H** (to see what is underneath) |
| Done / skip / previous / next | **Enter** / **S** / **A** or ← / **D** or → |

What to label:

- Every vehicle whose box you can draw with confidence, including partly hidden ones and the parts of
  vehicles cut by the frame edge.
- `van`: minibuses and panel vans (e.g. 16-seat Ford Transit). Ordinary cars, SUVs and taxis are `car`.
- `others`: trucks and anything else motorised that is not a car, van, bus or motorcycle.
- `motorcycle`: the bike and its rider(s) in one box.
- `ignore`: an area where vehicles are too small or crowded to box one by one (e.g. the far end of the
  road). It is blackened for training, so nothing inside is learnt as background. Prefer boxing over
  ignoring wherever you can.
- **S** (skip) a frame that is unusable (frozen, corrupt, camera moving).

Every change is saved within half a second to `labels/<id>.json`; the server can be stopped at any time.

## 3. Export

```powershell
.\.venv\Scripts\python.exe perception\finetune\export_coco.py hanoi --project data\labeling\hanoi-v1 --out perception\models\hanoi_coco
.\.venv\Scripts\python.exe perception\finetune\export_coco.py yolo --src data\external\vehicle-danang-2025 --out perception\models\danang_coco
```

Only frames marked done are exported. Each image carries the training adapter's partial-label fields:
Hanoi labels cover car, bus, van, others, motorcycle (`car_or_van: false`); Da Nang covers car, bus,
others (truck), motorcycle with vans inside `car` (`car_or_van: true`). Person and traffic light are not
labelled in either, so they are masked in training rather than learnt as background.

The Da Nang dataset is downloaded by hand from Kaggle (`keyshiftf/vehicle-danang-2025`) into
`data\external\vehicle-danang-2025`; the converter accepts `<split>/images + <split>/labels` and
`images/<split> + labels/<split>` layouts and reads class names from `data.yaml`.

## 4. Fine-tune locally

```powershell
$env:PYTHONUTF8 = 1
.\.venv\Scripts\python.exe perception\finetune\finetune.py --dataset perception\models\hanoi_coco --out runs\finetune\hanoi-v1
```

Starts from the deployed `rfdetr_nano_traffic7_bundle` weights (not COCO) at a lower learning rate
(5e-5, 25 epochs, batch 2 × accumulation 4 at 512 px: about 1.6 GB of GPU memory and 2.5 min per epoch for 140
frames on a GTX 1650). The partial-label adapter masks person and traffic light on Hanoi frames. The test split is
scored with the starting weights and again with the best new weights; both are in `heldout_metrics.json`.
The Hanoi test labels began as the old model's pre-labels, so the "before" score is optimistic.
`PYTHONUTF8=1` is needed on Windows because rfdetr prints its metrics tables in Unicode.

Needs `rfdetr[train]==1.11.1` extras. Install them without `opencv-python-headless`, which would replace the GUI
`opencv-python` that the live viewers use.

## Not done yet

Fine-tuning here uses only the exported folder, so nothing rehearses the BDD100K / UA-DETRAC data the model was
first trained on. Mixing `hanoi_coco` / `danang_coco` into the prepared dataset for the Kaggle notebook (the
adapter already accepts the `hanoi` and `danang` sources) would guard against drift on person / traffic light.
