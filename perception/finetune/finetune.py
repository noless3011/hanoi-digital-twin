"""Fine-tune the current RF-DETR Nano on an exported fine-tuning dataset (export_coco.py) on this machine.

Training starts from the deployed seven-class weights, not from COCO, with a lower learning rate, so the
model adapts to the new cameras without relearning everything. Classes a source does not label (person and
traffic light for Hanoi frames) are masked by the partial-label adapter, so they are not learnt as background.

Before training, the starting weights are scored on the dataset's test split; afterwards the best new weights
are scored the same way, and both results land in <out>/heldout_metrics.json.

    .venv\\Scripts\\python.exe perception\\finetune\\finetune.py --dataset perception\\models\\hanoi_coco --out runs\\finetune\\hanoi-v1

The new weights are <out>/checkpoint_best_total.pth. They replace nothing; copy them into a model bundle
once the comparison is satisfactory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perception" / "training" / "helpers"))
sys.path.insert(0, str(ROOT / "perception"))

CLASS_NAMES = ["car", "bus", "van", "others", "person", "motorcycle", "traffic light"]


def evaluate(weights, dataset, batch_size, workers):
    from rfdetr.detr import RFDETR
    model = RFDETR.from_checkpoint(str(weights), device="cuda", compile=False)
    metrics = model.evaluate(split="test", dataset_dir=str(dataset), dataset_file="roboflow",
                             batch_size=batch_size, num_workers=workers, device="cuda", amp_dtype="auto")
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}


def main(argv=None):
    import run_rfdetr
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder from export_coco.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=run_rfdetr.DEFAULT_MODEL, help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=2, help="Images per step (2 fits a 4 GB GPU at 512 px)")
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-encoder", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    from rfdetr import RFDETRNano
    from partial_labels import install_partial_label_support
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    if args.out.exists() and any(args.out.glob("*.pth")):
        raise SystemExit(f"{args.out} already has weights; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=True)

    install_partial_label_support(args.dataset)
    before = evaluate(args.weights, args.dataset, args.batch, args.workers)
    print("starting weights, test split:", json.dumps(before, indent=1), flush=True)
    torch.cuda.empty_cache()

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["gpu"] = torch.cuda.get_device_name(0)
    (args.out / "finetune_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    model = RFDETRNano(pretrain_weights=str(args.weights), resolution=512, num_classes=len(CLASS_NAMES),
                       device="cuda", compile=False, cuda_graphs=False)
    model.train(dataset_dir=str(args.dataset), dataset_file="roboflow", output_dir=str(args.out),
                epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.accum,
                lr=args.lr, lr_encoder=args.lr_encoder, num_workers=args.workers, class_names=CLASS_NAMES,
                multi_scale=False, expanded_scales=False, amp_dtype="auto", checkpoint_interval=5,
                seed=args.seed, devices=1, run_test=False, tensorboard=True, wandb=False, progress_bar="tqdm")
    del model
    torch.cuda.empty_cache()

    best = args.out / "checkpoint_best_total.pth"
    after = evaluate(best, args.dataset, args.batch, args.workers)
    (args.out / "heldout_metrics.json").write_text(json.dumps({"before": before, "after": after}, indent=1), encoding="utf-8")
    for name in ("class_names.json",):
        src = args.weights.parent / name
        if src.exists():
            shutil.copyfile(src, args.out / name)
    print("\n           before   after")
    for k in ("test/mAP_50_95", "test/mAP_50", "test/precision", "test/recall"):
        print(f"{k:16} {before.get(k, float('nan')):.3f}   {after.get(k, float('nan')):.3f}")
    print(f"New weights: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
