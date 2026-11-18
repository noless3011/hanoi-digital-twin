"""Source-aware RF-DETR 1.11.1 losses and metrics for mixed traffic labels.

UA-DETRAC does not annotate people, motorcycles, or traffic lights. BDD100K
annotates vans as cars. A raw concatenation would therefore penalize
correct detections and teach conflicting car/van labels. This adapter masks
unknown classes and supervises external car boxes with P(car OR van). The
original seven-class outputs and saved model weights remain unchanged.
"""

from __future__ import annotations

import contextvars
import functools
import importlib.metadata
import json
from pathlib import Path

import torch


PARTIAL_CLASS_NAMES = ("car", "bus", "van", "others", "person", "motorcycle", "traffic light")
SUPPORTED_VERSION = "1.11.1"
MASK_LOGIT = -30.0
_CONTEXT = contextvars.ContextVar("traffic_partial_label_targets", default=None)
# Colab users may rerun the embedded helper cell. Keep the existing handle so
# the next install can restore the original methods before replacing policies.
_INSTALLED = globals().get("_INSTALLED")


def load_image_policies(dataset_dir):
    """Read globally unique image IDs and category-ID supervision metadata.

    Each image must provide ``source``, ``known_class_ids`` (COCO category IDs,
    not model indices), and ``car_or_van``. Category IDs map to model indices in
    ascending order, exactly as RF-DETR's custom COCO loader does.
    """
    root = Path(dataset_dir)
    policies = {}
    expected_categories = None
    for split in ("train", "valid", "test"):
        path = root / split / "_annotations.coco.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing mixed-dataset annotations: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        categories = sorted(document["categories"], key=lambda item: int(item["id"]))
        names = tuple(item["name"] for item in categories)
        if names != PARTIAL_CLASS_NAMES:
            raise ValueError(f"{path}: expected classes in this order: {PARTIAL_CLASS_NAMES}")
        signature = tuple((int(item["id"]), item["name"]) for item in categories)
        if expected_categories is not None and signature != expected_categories:
            raise ValueError(f"{path}: category IDs differ from the training split")
        expected_categories = signature
        cat_to_label = {category_id: index for index, (category_id, _) in enumerate(signature)}
        local_ids = set()
        for image in document["images"]:
            image_id = int(image["id"])
            if image_id in policies:
                raise ValueError(f"{path}: duplicate global image ID {image_id}")
            source = image.get("source")
            if source not in {"ua_detrac", "coco", "bdd100k"}:
                raise ValueError(f"Image {image_id}: missing or unsupported source {source!r}")
            known_ids = image.get("known_class_ids")
            if not isinstance(known_ids, list) or not known_ids:
                raise ValueError(f"Image {image_id}: known_class_ids must contain category IDs")
            try:
                known_labels = tuple(sorted({cat_to_label[int(value)] for value in known_ids}))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Image {image_id}: invalid known_class_ids") from exc
            coarse_car = image.get("car_or_van")
            if not isinstance(coarse_car, bool):
                raise ValueError(f"Image {image_id}: car_or_van must be a boolean")
            if coarse_car and (0 not in known_labels or 2 in known_labels):
                raise ValueError(f"Image {image_id}: coarse car labels require car known and van unknown")
            if source == "ua_detrac" and coarse_car:
                raise ValueError(f"Image {image_id}: UA-DETRAC has separate car and van labels")
            if source in {"coco", "bdd100k"} and 0 in known_labels and not coarse_car:
                raise ValueError(f"Image {image_id}: external car labels must supervise car OR van")
            policies[image_id] = {
                "source": source,
                "known_labels": known_labels,
                "car_or_van": coarse_car,
            }
            local_ids.add(image_id)
        for annotation in document["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id not in local_ids:
                raise ValueError(f"{path}: annotation references missing image {image_id}")
            label = cat_to_label.get(int(annotation["category_id"]))
            if label not in policies[image_id]["known_labels"]:
                raise ValueError(f"Image {image_id}: annotation uses an unknown/unsupervised class")
    if not policies:
        raise ValueError("Mixed dataset contains no images")
    return policies


def _policies_for_targets(targets, registry):
    policies = []
    for target in targets:
        value = target.get("image_id")
        if value is None:
            raise ValueError("Partial-label training requires each target's image_id")
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("Each target image_id must contain exactly one value")
            image_id = int(value.item())
        else:
            image_id = int(value)
        try:
            policies.append(registry[image_id])
        except KeyError as exc:
            raise ValueError(f"Image {image_id} is absent from the installed dataset registry") from exc
    return policies


def transform_logits(logits, policies, *, for_metrics=False):
    """Make virtual native-dataset logits without changing the original tensor.

    logit(P(car OR van)) = logsumexp(car, van, car + van). Unknown
    foreground slots are constant, so they have zero classification gradient.
    RF-DETR's eighth background slot keeps its normal negative supervision in
    the loss; it is suppressed and filtered from detection metrics.
    """
    if logits.ndim != 3 or logits.shape[0] != len(policies) or logits.shape[2] != 8:
        raise ValueError("RF-DETR 1.11.1 traffic logits must have shape (batch, queries, 8)")
    rows = []
    for index, policy in enumerate(policies):
        row = logits[index]
        known = set(policy["known_labels"])
        columns = []
        for label in range(8):
            if label == 0 and policy["car_or_van"]:
                # FP32 avoids overflow in car+van under mixed precision.
                car, van = row[:, 0].float(), row[:, 2].float()
                pooled = torch.logsumexp(torch.stack((car, van, car + van), dim=-1), dim=-1)
                columns.append(pooled.to(dtype=row.dtype))
            elif label in known or (label == 7 and not for_metrics):
                columns.append(row[:, label])
            else:
                columns.append(torch.full_like(row[:, label], MASK_LOGIT))
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=0)


def transform_outputs(outputs, policies, *, for_metrics=False):
    """Apply the policy to the final, auxiliary, and encoder output dictionaries."""
    result = dict(outputs)
    result["pred_logits"] = transform_logits(outputs["pred_logits"], policies, for_metrics=for_metrics)
    if "aux_outputs" in outputs:
        result["aux_outputs"] = [
            transform_outputs(item, policies, for_metrics=for_metrics) for item in outputs["aux_outputs"]
        ]
    if "enc_outputs" in outputs:
        result["enc_outputs"] = transform_outputs(outputs["enc_outputs"], policies, for_metrics=for_metrics)
    return result


class PartialLabelSupport:
    """Reversible, version-pinned runtime patches; no model state is added."""

    def __init__(self, registry, classes):
        self.registry = registry
        self._patches = []
        criterion_class, postprocess_class, module_class, callback_class = classes
        original_criterion = criterion_class.forward
        original_postprocess = postprocess_class.forward

        @functools.wraps(original_criterion)
        def criterion_forward(criterion, outputs, targets, num_boxes=None):
            if criterion.num_classes != len(PARTIAL_CLASS_NAMES) + 1:
                raise ValueError("Partial-label support requires seven foreground classes plus the background slot")
            policies = _policies_for_targets(targets, self.registry)
            virtual_outputs = transform_outputs(outputs, policies)
            return original_criterion(criterion, virtual_outputs, targets, num_boxes=num_boxes)

        @functools.wraps(original_postprocess)
        def postprocess_forward(postprocess, outputs, target_sizes, score_threshold=None):
            targets = _CONTEXT.get()
            if targets is None:
                # Standard seven-class inference must use the unmodified heads.
                return original_postprocess(postprocess, outputs, target_sizes, score_threshold=score_threshold)
            policies = _policies_for_targets(targets, self.registry)
            virtual_outputs = transform_outputs(outputs, policies, for_metrics=True)
            results = original_postprocess(
                postprocess, virtual_outputs, target_sizes, score_threshold=score_threshold
            )
            filtered = []
            for result, policy in zip(results, policies):
                labels = result["labels"]
                keep = torch.zeros_like(labels, dtype=torch.bool)
                for label in policy["known_labels"]:
                    keep |= labels == label
                filtered.append({
                    key: value[keep] if torch.is_tensor(value) and value.ndim and value.shape[0] == len(labels) else value
                    for key, value in result.items()
                })
            return filtered

        self._patch(criterion_class, "forward", criterion_forward)
        self._patch(postprocess_class, "forward", postprocess_forward)
        for method in ("training_step", "validation_step", "test_step"):
            self._patch_step(module_class, method)
        # eval_base_model=True runs an additional EMA postprocess in this hook.
        self._patch_step(callback_class, "on_validation_batch_end", callback=True)

    def _patch(self, cls, name, wrapper):
        original = getattr(cls, name)
        setattr(cls, name, wrapper)
        self._patches.append((cls, name, original, wrapper))

    def _patch_step(self, cls, name, *, callback=False):
        original = getattr(cls, name)

        @functools.wraps(original)
        def step(instance, *args, **kwargs):
            # Bound module step: batch is arg0. Bound callback hook: batch arg3.
            batch = kwargs.get("batch")
            if batch is None:
                batch = args[3 if callback else 0]
            token = _CONTEXT.set(batch[1])
            try:
                return original(instance, *args, **kwargs)
            finally:
                _CONTEXT.reset(token)

        self._patch(cls, name, step)

    def uninstall(self):
        """Restore the original methods after this notebook's training/evaluation."""
        global _INSTALLED
        for cls, name, original, wrapper in reversed(self._patches):
            if getattr(cls, name) is wrapper:
                setattr(cls, name, original)
        self._patches.clear()
        if _INSTALLED is self:
            _INSTALLED = None


def install_partial_label_support(dataset_dir):
    """Install before model.train()/evaluate(); repeat calls safely replace policies.

    Intended for the notebook's single-process Colab GPU training. These Python
    metadata lookups require eager criterion execution; use compile=False,
    cuda_graphs=False, and ordinary COCO loading rather than packed webdataset.
    """
    global _INSTALLED
    installed_version = importlib.metadata.version("rfdetr")
    if installed_version != SUPPORTED_VERSION:
        raise RuntimeError(
            f"Partial-label adapter supports rfdetr=={SUPPORTED_VERSION}; found {installed_version}. "
            "Reinstall the notebook's pinned dependencies and restart the runtime."
        )
    registry = load_image_policies(dataset_dir)
    from rfdetr.models.criterion import SetCriterion
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
    from rfdetr.training.module_model import RFDETRModelModule

    if _INSTALLED is not None:
        _INSTALLED.uninstall()
    _INSTALLED = PartialLabelSupport(
        registry, (SetCriterion, PostProcess, RFDETRModelModule, COCOEvalCallback)
    )
    return _INSTALLED
