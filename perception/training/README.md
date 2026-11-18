# Prepare locally, train in Kaggle or Colab

Prepare the data once on your machine with [prepare_traffic_dataset.py](prepare_traffic_dataset.py), using the locations of your extracted BDD100K and UA-DETRAC folders. Then use [train_rfdetr_nano_ua_detrac.ipynb](train_rfdetr_nano_ua_detrac.ipynb) for GPU training, metrics, checkpoint resume, evaluation, and result downloads. The notebook consumes the prepared folder directly; conversion helpers stay on your machine.

## Prepare the dataset

Extract BDD100K and UA-DETRAC yourself before running the script. Pass one dataset root for each with `--bdd-dir` and `--ua-dir`; nested wrapper directories are supported. Both flags are required together, and discovery inside these roots uses extracted files only. Preparation uses only these two datasets and runs fully offline:

| Source | Local inputs |
| --- | --- |
| BDD100K (`--bdd-dir`) | An extracted dataset containing native `images/100k/train`, `images/100k/val`, and detection JSONs under `labels`, or Supervisely `train/ann`, `val/ann`, `train/img`, `val/img`. Put images and labels under the same dataset root. |
| UA-DETRAC (`--ua-dir`) | An extracted dataset containing image sequences such as `images/MVI_*` and official TRAIN XMLs, such as `DETRAC-Train-Annotations-XML/MVI_*.xml`. Put images and XMLs under the same dataset root. TEST annotations and MOT toolkit files are excluded. |

Run from the repository root using the existing environment (only Pillow and Python's standard library are needed):

```powershell
.\.venv\Scripts\python.exe perception\training\prepare_traffic_dataset.py --bdd-dir "D:\datasets\BDD100K" --ua-dir "D:\datasets\UA-DETRAC" --output-dir "D:\datasets\traffic_clean" --archive
```

Replace the input folder paths with your extracted dataset locations, including folders under `perception/models` if you extracted there. `--output-dir` sets the final prepared folder location; the older `--output` spelling is also accepted. Without either, the default is `perception/models/traffic7_coco`.

`--archive` optionally packages the finished dataset into a ZIP in addition to keeping the prepared folder. With the example above, the ZIP is `D:\datasets\traffic_clean.zip`. Omit `--archive` to save the extra disk space and packaging time, or supply a ZIP destination such as `--archive "D:\exports\traffic_clean.zip"`. It does not unpack your input datasets. A suitable extracted layout is:

```text
BDD100K/
  train/ann/ + train/img/    # Supervisely export
  val/ann/ + val/img/
  # or native images/100k/{train,val}/ + labels/*train*.json, *val*.json
UA-DETRAC/
  images/MVI_20011/img00001.jpg
  images/MVI_20012/img00001.jpg
  ...
  DETRAC-Train-Annotations-XML/MVI_20011.xml
  DETRAC-Train-Annotations-XML/MVI_20012.xml
  ...
```

Raw archives inside the input roots are ignored; extract images and labels before running. No dataset downloads occur.

`--help` lists folder inputs, per-source caps, sampling controls, and output locations. Defaults preserve the previous notebook recipe: up to 6,000 training and 600 validation/600 internal test images for BDD100K, every fifth UA-DETRAC frame capped at 100 per sequence, seed 42. A cap of `0` removes that cap. All 60 official UA-DETRAC TRAIN sequences are expected; `--expected-ua-sequences 0` permits a custom subset with at least three sequences.

For a small preparation check, use a new output directory and smaller caps:

```powershell
.\.venv\Scripts\python.exe perception\training\prepare_traffic_dataset.py `
  --bdd-dir "D:\datasets\BDD100K" --ua-dir "D:\datasets\UA-DETRAC" `
  --output-dir perception\models\traffic7_smoke `
  --bdd-train-cap 100 --bdd-valid-cap 30 --bdd-test-cap 30 `
  --max-frames-per-sequence 20 --archive
```

The script reads the two extracted input folders, converts XML/native JSON/Supervisely labels, masks UA-DETRAC ignored regions, clips boxes, creates disjoint splits, and merges the seven-class dataset in COCO annotation format. It validates the result and computes content fingerprints locally. Your input folders are preserved. A matching completed output and upload ZIP can be reused after content verification; a different configuration or an incomplete/untracked output requires a fresh `--output`. Existing ZIPs with different contents are refused. Reusable intermediate files live under `perception/models/.traffic7_work`; retain them if you expect to rerun preparation.

The output uses COCO **annotation format**; no images from the COCO dataset are included.

Default output:

```text
perception/models/traffic7_coco/
  manifest.json
  train/  _annotations.coco.json + images
  valid/  _annotations.coco.json + images
  test/   _annotations.coco.json + images
perception/models/traffic7_coco.zip  # with --archive
```

The ZIP contains the complete prepared folder under its directory name (`traffic7_coco/` by default). Upload and extract it before starting training, or use a platform upload that exposes the archive's extracted contents. The notebook expects an existing directory containing `manifest.json` and the three split folders. It does not unpack raw or prepared archives.

## Run the notebook

**Kaggle:** import the notebook, select a GPU, enable Internet for installation/pretrained weights, and attach the prepared dataset using Add Input. **Colab:** upload the notebook, select a GPU runtime, and put the prepared folder in Drive. The notebook remains self-contained; no helper files or repository clone are needed.

Set `PREPARED_DATASET_DIR` to the folder containing `manifest.json`. Kaggle can automatically discover the single prepared dataset under its input root; multiple prepared datasets require an explicit path. Colab defaults to `MyDrive/traffic7_coco` when the path is empty. The notebook reads that folder as supplied and checks manifest/category/count/annotation-fingerprint consistency. It uses the locally computed image-content fingerprint without scanning every image's bytes again. Keep the prepared files unchanged after conversion.

Training defaults to 30 total epochs at 512 pixels, batch 4, accumulation 4. Use a new `RUN_NAME` when changing the prepared dataset or training settings. For a quick GPU check use 2 epochs. TensorBoard/inline plots track training and validation; evaluation reloads the selected best weights and saves held-out metrics and a prediction preview.

Kaggle saves runs under `/kaggle/working/traffic_training/runs/<RUN_NAME>`; Colab saves runs in Drive under `MyDrive/traffic_training/runs/<RUN_NAME>`. Save a Kaggle notebook version with outputs to retain full checkpoints. To resume, attach saved outputs and set `RESUME_RUN_DIR` to the run folder containing `last.ckpt` and `run_config.json`, keeping dataset and settings unchanged. `EPOCHS` is the total target epoch count. The notebook restores run artifacts to writable storage and checks the saved training signature.

Download `rfdetr_nano_traffic7_bundle.zip` using the final cell. It contains `checkpoint_best_total.pth`, class names, run/data configuration, runtime requirements, held-out metrics, prediction preview, inference script, and the training adapter. Full resume checkpoints and TensorBoard logs remain in the run folder. Keep `last.ckpt` separately for continuing training.

## Labels and evaluation

The fixed class order is `car`, `bus`, `van`, `others`, `person`, `motorcycle`, `traffic light`. `person` includes pedestrians and riders; Supervisely `motor` maps to motorcycle. `others` retains UA-DETRAC's vehicle category and receives BDD trucks. Traffic lights are signal objects without color classes.

The notebook retains the source-aware training/evaluation adapter: UA-DETRAC does not supply the three extra classes, and BDD broad car labels supervise car-or-van. UA-DETRAC supplies the separate van labels. These policies remain in each prepared image's metadata. Training uses pinned `rfdetr==1.11.1`, eager single-process training on GPU 0, and the original seven-class outputs for inference.

UA-DETRAC is split by sequence. BDD official train images remain in training; their labeled validation images are divided into disjoint internal validation/test candidates before sampling. BDD images with target crowd/ignore annotations are excluded; UA-DETRAC ignored regions are masked. These are custom detection holdouts, not official source benchmarks or tracking metrics. TrackTrack remains a subsequent association stage.

## Maintaining the notebook

The generator embeds only training/runtime support and the source-aware adapter:

```powershell
.\.venv\Scripts\python.exe perception\training\build_training_notebook.py
.\.venv\Scripts\python.exe -m unittest discover -s perception/training/tests -p 'test_*.py'
```

Synthetic tests cover archive/folder preparation, label mapping, masking, sampling, leakage, fingerprints, prepared-data notebook loading, resume guards, training orchestration, and deployment bundles. They do not measure model accuracy or replace a full real-data GPU run.
