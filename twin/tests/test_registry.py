"""Checks that the tools reach a behaviour model only through the registry, so a new one plugs in."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from twin import scenarios as sc
from twin.behaviour import (BehaviourModel, ChoiceModel, ParamSet, PriorityModel, Scene, Spec, default_params,
                            make_model, model_names, register, registry)
from twin.engine import Arrival, Simulation
from twin.examples.follower_model import FollowerModel

FOLLOWER = "twin.examples.follower_model:FollowerModel"


class RegistryTests(unittest.TestCase):
    def test_built_in_models_by_name(self):
        self.assertLessEqual({"choice", "priority"}, set(model_names()))
        self.assertIsInstance(make_model(default_params("choice")), ChoiceModel)
        self.assertIsInstance(make_model(default_params("priority")), PriorityModel)

    def test_unknown_names_say_what_exists(self):
        with self.assertRaisesRegex(KeyError, "choice"):
            registry.get("nonsense")
        with self.assertRaisesRegex(KeyError, "Cannot load"):
            registry.get("twin.examples.follower_model:Missing")

    def test_a_name_belongs_to_one_class(self):
        with self.assertRaises(ValueError):
            register("choice")(type("Other", (BehaviourModel,), {}))

    def test_the_scene_shapes_the_priority_parameters(self):
        scene = Scene(network=sc.four_way(), route_arrivals=np.array([50, 1, 30, 0]), turns=2)
        names = default_params("priority", scene).names
        self.assertIn("v_r0_k0", names)
        self.assertIn("v_r2_k0", names)
        self.assertNotIn("v_r1_k0", names)       # 1 of 81 arrivals: no profile of its own
        self.assertIn("turn_2", names)
        self.assertEqual(default_params("choice", scene).names, ParamSet().names)

    def test_a_model_registered_by_decorator(self):
        @register("test-crawl")
        class Crawl(BehaviourModel):
            SPECS = (Spec("speed", 2.0, 0.5, 5.0, False, 1, "speed of everyone (m/s)"),)

            def sample_driver(self, kind, arrival, rng, right, left, width):
                return {"v": self.params.values["speed"], "d": 0.0}

            def decide(self, view):
                from twin.behaviour import Decision
                return Decision(acc=np.zeros(len(view.slots)), vd=np.zeros(len(view.slots)))

        try:
            params = default_params("test-crawl").with_values({"speed": 4.0})
            self.assertEqual(params.model, "test-crawl")
            res = Simulation(sc.straight(length=100.0), make_model(params), [Arrival(0, "motorcycle", 0)], seed=1).run(60)
            self.assertAlmostEqual(res.vehicles[0]["t_exit"], 25.0, delta=1.0)
        finally:
            registry._MODELS.pop("test-crawl")


class PluggedInModelTests(unittest.TestCase):
    """twin/examples/follower_model.py, which nothing registers, used by its module:Class path."""

    def test_parameters_and_files(self):
        params = default_params(FOLLOWER)
        self.assertEqual(params.names, tuple(s.name for s in FollowerModel.SPECS))
        self.assertEqual(params.model, FOLLOWER)
        x = params.to_unit(params.names)
        self.assertTrue(all(0 <= u <= 1 for u in x))
        with tempfile.TemporaryDirectory() as tmp:
            params.with_values({"T": 1.7}).save(Path(tmp) / "p.json")
            loaded = ParamSet.load(Path(tmp) / "p.json")
        self.assertEqual(loaded.values["T"], 1.7)
        self.assertIsInstance(make_model(loaded), FollowerModel)

    def test_follows_without_touching(self):
        # A fast motorcycle behind a slow one on one line: it closes up and stays behind.
        arrivals = [Arrival(0, "motorcycle", 0, v0=4.0, d=0.0), Arrival(3, "motorcycle", 0, v0=10.0, d=0.0)]
        res = Simulation(sc.straight(length=300.0), make_model(default_params(FOLLOWER)), arrivals, seed=1).run(120)
        summary = res.summary()
        self.assertEqual((summary["exited"], summary["contact_pairs"]), (2, 0))
        first, second = (v["t_exit"] for v in sorted(res.vehicles, key=lambda v: v["t_spawn"]))
        self.assertGreater(second, first)

    def test_a_parameter_changes_the_outcome(self):
        net, arrivals = sc.straight(length=300.0), sc.poisson(sc.straight(length=300.0), 900, 60, seed=2)

        def mean_time(values):
            res = Simulation(net, make_model(default_params(FOLLOWER).with_values(values)), arrivals, seed=1).run(60, until_empty=True)
            return float(np.mean([v["t_exit"] - v["t_spawn"] for v in res.vehicles]))
        self.assertGreater(mean_time({"v0": 5.0}), mean_time({"v0": 10.0}) + 5.0)


if __name__ == "__main__":
    unittest.main()


class CoreChoiceTests(unittest.TestCase):
    def test_the_core_model_is_the_choice_model_with_fewer_free_parameters(self):
        core = default_params("choice-core")
        self.assertEqual(core.model, "choice-core")
        self.assertNotIn("vlat", core.names)
        self.assertIn("w_risk", core.names)
        values = {"a": 3.0, "w_risk": 2.0}
        net = sc.four_way()
        arrivals = [Arrival(1.5 * i, "motorcycle", i % 4) for i in range(30)]
        a = Simulation(net, make_model(core.with_values(values)), arrivals, seed=3).run(60)
        b = Simulation(net, make_model(ParamSet().with_values(values)), arrivals, seed=3).run(60)
        self.assertEqual([v["t_exit"] for v in a.vehicles], [v["t_exit"] for v in b.vehicles])
        with tempfile.TemporaryDirectory() as d:
            core.with_values(values).save(Path(d) / "p.json")
            loaded = ParamSet.load(Path(d) / "p.json")
        self.assertEqual(loaded.model, "choice-core")
        self.assertEqual(loaded.of("car"), core.with_values(values).of("car"))
