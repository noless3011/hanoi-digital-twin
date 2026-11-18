"""Verify mixed-label gradients and native-dataset metric policies."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from rfdetr.models.criterion import SetCriterion
    from rfdetr.models.matcher import HungarianMatcher
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.models.lwdetr import build_criterion_from_config
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
except ImportError:
    torch = None


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if torch is not None:
    spec = importlib.util.spec_from_file_location("partial_labels", NOTEBOOK_ROOT / "helpers/partial_labels.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)


@unittest.skipIf(torch is None, "RF-DETR/PyTorch environment required")
class PartialLabelTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            1: {"source": "ua_detrac", "known_labels": (0, 1, 2, 3), "car_or_van": False},
            2: {"source": "coco", "known_labels": (0, 1, 3, 4, 5, 6), "car_or_van": True},
            3: {"source": "bdd100k", "known_labels": (0, 1, 3, 4, 5, 6), "car_or_van": True},
        }

        class Module:
            def __init__(self, postprocess):
                self.postprocess = postprocess

            def validation_step(self, batch, batch_idx):
                outputs, targets = batch
                sizes = torch.stack([target["orig_size"] for target in targets])
                return {"results": self.postprocess(outputs, sizes), "targets": targets}

            training_step = validation_step
            test_step = validation_step

        class Callback:
            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                # Same path used by the independently forwarded EMA model.
                raw_outputs, targets = batch
                sizes = torch.stack([target["orig_size"] for target in targets])
                return pl_module.postprocess(raw_outputs, sizes)

        self.Module = Module
        self.Callback = Callback
        self.adapter = helper.PartialLabelSupport(
            self.registry, (SetCriterion, PostProcess, Module, Callback)
        )

    def tearDown(self):
        self.adapter.uninstall()

    def targets(self):
        return [
            {"image_id": torch.tensor([1]), "labels": torch.tensor([1]),
             "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "orig_size": torch.tensor([100, 100])},
            {"image_id": torch.tensor([2]), "labels": torch.tensor([0]),
             "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "orig_size": torch.tensor([100, 100])},
        ]

    def outputs(self):
        def layer():
            return {
                "pred_logits": torch.full((2, 2, 8), -2.0, requires_grad=True),
                "pred_boxes": torch.tensor([
                    [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]],
                    [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]],
                ], requires_grad=True),
            }
        result = layer()
        result["aux_outputs"] = [layer(), layer()]
        result["enc_outputs"] = layer()
        return result

    def criterion(self):
        criterion, _ = build_criterion_from_config(
            RFDETRNanoConfig(num_classes=7, group_detr=1),
            TrainConfig(dataset_dir=".", accelerator="cpu"),
        )
        self.assertEqual(criterion.num_classes, 8)
        self.assertTrue(criterion.ia_bce_loss)
        return criterion

    def assert_gradients(self, outputs):
        for layer in [outputs, *outputs["aux_outputs"], outputs["enc_outputs"]]:
            gradient = layer["pred_logits"].grad
            self.assertIsNotNone(gradient)
            self.assertEqual(gradient[0, :, 4:7].abs().sum().item(), 0)
            # The broad car positive trains both finer vehicle heads.
            self.assertLess(gradient[1, 0, 0].item(), 0)
            self.assertLess(gradient[1, 0, 2].item(), 0)
            # Both heads get negative supervision on an unmatched query.
            self.assertGreater(gradient[1, 1, 0].item(), 0)
            self.assertGreater(gradient[1, 1, 2].item(), 0)
            # The extra background slot retains stock negative supervision.
            self.assertGreater(gradient[:, :, 7].sum().item(), 0)

    def test_real_criterion_main_aux_encoder_have_safe_gradients(self):
        outputs = self.outputs()
        raw = outputs["pred_logits"].detach().clone()
        losses = self.criterion()(outputs, self.targets())
        self.assertIn("loss_ce_enc", losses)
        self.assertIn("loss_ce_1", losses)
        sum(value for name, value in losses.items() if name.startswith("loss_ce")).backward()
        self.assert_gradients(outputs)
        self.assertTrue(torch.equal(raw, outputs["pred_logits"]))

    def test_real_layer_batched_loss_kernel_has_safe_gradients(self):
        outputs = self.outputs()
        criterion = self.criterion()
        # The stock CUDA fast path resolves to this same eager tensor kernel
        # on CPU. Force its dispatch to test layer batching without compilation.
        with patch.object(criterion, "_can_batch_detection_losses", return_value=True):
            losses = criterion(outputs, self.targets(), num_boxes=2.0)
        sum(value for name, value in losses.items() if name.startswith("loss_ce")).backward()
        self.assert_gradients(outputs)

    def test_unknown_ua_logits_do_not_change_losses_or_matching(self):
        outputs = self.outputs()
        criterion = self.criterion()
        before = criterion(outputs, self.targets())
        with torch.no_grad():
            outputs["pred_logits"][0, :, 4:7] = 80
            for layer in [*outputs["aux_outputs"], outputs["enc_outputs"]]:
                layer["pred_logits"][0, :, 4:7] = 80
        after = criterion(outputs, self.targets())
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_car_union_probability_identity_and_no_input_mutation(self):
        logits = torch.randn(3, 6, 8, requires_grad=True)
        original = logits.detach().clone()
        transformed = helper.transform_logits(logits, list(self.registry.values()))
        car, van = logits[1:, :, 0].sigmoid(), logits[1:, :, 2].sigmoid()
        expected = 1 - (1 - car) * (1 - van)
        self.assertTrue(torch.allclose(transformed[1:, :, 0].sigmoid(), expected, atol=1e-6))
        self.assertTrue(torch.equal(original, logits))
        self.assertTrue(torch.equal(transformed[0, :, :4], logits[0, :, :4]))

    def test_native_metric_masks_before_topk_and_inference_stays_raw(self):
        outputs = self.outputs()
        with torch.no_grad():
            outputs["pred_logits"][0, :, 4:] = 80  # all highest raw UA scores are unknown
            outputs["pred_logits"][1, :, 0] = -10
            outputs["pred_logits"][1, :, 2] = 8  # confident van -> coarse external car
            outputs["pred_logits"][1, :, 7] = 80
        postprocess = PostProcess(num_select=2)
        module = self.Module(postprocess)
        batch = (outputs, self.targets())
        for method in ("training_step", "validation_step", "test_step"):
            results = getattr(module, method)(batch, 0)["results"]
            self.assertEqual(len(results[0]["labels"]), 2)
            self.assertTrue(set(results[0]["labels"].tolist()) <= {0, 1, 2, 3})
            self.assertEqual(results[1]["labels"].tolist(), [0, 0])
            self.assertGreater(results[1]["scores"].min().item(), 0.99)
        ema_results = self.Callback().on_validation_batch_end(None, module, None, batch, 0)
        self.assertEqual(ema_results[1]["labels"].tolist(), [0, 0])
        # No context means normal inference, including exactly the original logits.
        raw_results = postprocess(outputs, torch.tensor([[100, 100], [100, 100]]))
        self.assertTrue(set(raw_results[0]["labels"].tolist()).isdisjoint({0, 1, 2, 3}))
        self.assertEqual(raw_results[1]["labels"].tolist(), [7, 7])

    def test_metric_context_is_reset_after_error(self):
        module = self.Module(PostProcess(num_select=2))
        with self.assertRaises(ValueError):
            module.validation_step((self.outputs(), [{"image_id": torch.tensor([99]),
                "orig_size": torch.tensor([100, 100])}] * 2), 0)
        self.assertIsNone(helper._CONTEXT.get())

    def test_unknown_id_or_head_shape_fails_closed(self):
        outputs = self.outputs()
        targets = self.targets()
        targets[0]["image_id"] = torch.tensor([999])
        with self.assertRaisesRegex(ValueError, "absent"):
            self.criterion()(outputs, targets)
        with self.assertRaisesRegex(ValueError, "shape"):
            helper.transform_logits(torch.zeros(2, 2, 7), list(self.registry.values())[:2])

    def test_registry_checks_sources_classes_and_unique_split_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categories = [{"id": index + 1, "name": name} for index, name in enumerate(helper.PARTIAL_CLASS_NAMES)]
            for index, (split, policy) in enumerate(zip(("train", "valid", "test"), self.registry.values()), 1):
                (root / split).mkdir()
                document = {"categories": categories, "annotations": [], "images": [{
                    "id": index, "source": policy["source"], "car_or_van": policy["car_or_van"],
                    "known_class_ids": [label + 1 for label in policy["known_labels"]],
                }]}
                (root / split / "_annotations.coco.json").write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(helper.load_image_policies(root), self.registry)
            path = root / "test/_annotations.coco.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["images"][0]["id"] = 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate global"):
                helper.load_image_policies(root)

    def test_install_replaces_registry_and_embedded_cell_rerun_keeps_handle(self):
        self.adapter.uninstall()
        original = SetCriterion.forward
        module_shim = types.ModuleType("rfdetr.training.module_model")
        module_shim.RFDETRModelModule = self.Module
        callback_shim = types.ModuleType("rfdetr.training.callbacks.coco_eval")
        callback_shim.COCOEvalCallback = self.Callback
        second_registry = {99: self.registry[1]}
        with patch.dict(sys.modules, {
            "rfdetr.training.module_model": module_shim,
            "rfdetr.training.callbacks.coco_eval": callback_shim,
        }), patch.object(helper, "load_image_policies", return_value=self.registry):
            first = helper.install_partial_label_support("first")
        try:
            self.assertIs(helper._INSTALLED, first)
            exec((NOTEBOOK_ROOT / "helpers/partial_labels.py").read_text(encoding="utf-8"), helper.__dict__)
            self.assertIs(helper._INSTALLED, first)
            with patch.dict(sys.modules, {
                "rfdetr.training.module_model": module_shim,
                "rfdetr.training.callbacks.coco_eval": callback_shim,
            }), patch.object(helper, "load_image_policies", return_value=second_registry):
                second = helper.install_partial_label_support("second")
            self.assertFalse(first._patches)
            self.assertIs(helper._INSTALLED, second)
            with self.assertRaisesRegex(ValueError, "absent"):
                self.criterion()(self.outputs(), self.targets())
            second.uninstall()
            self.assertIs(SetCriterion.forward, original)
            second.uninstall()  # teardown is idempotent too
        finally:
            if helper._INSTALLED is not None:
                helper._INSTALLED.uninstall()

    def test_unsupported_rfdetr_version_fails_before_install(self):
        with patch.object(helper.importlib.metadata, "version", return_value="1.12.0"):
            with self.assertRaisesRegex(RuntimeError, "supports rfdetr==1.11.1"):
                helper.install_partial_label_support("unused")


if __name__ == "__main__":
    unittest.main()
