"""Fast checks of the network, the engine with the constant-speed model, and track measurement."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from twin import scenarios as sc
from twin.behaviour import BehaviourModel, ConstantSpeedModel, Decision
from twin.engine import Arrival, Simulation
from twin.measure.image_score import wasserstein
from twin.measure.tracks import from_result, kalman_smooth, load, stitch
from twin.network import CROSS, MERGE


class NetworkTests(unittest.TestCase):
    def test_projection_round_trip(self):
        net = sc.four_way()
        for r in range(len(net.routes)):
            s = np.linspace(5, net.length[r] - 5, 9)
            d = np.linspace(-2, 2, 9)
            xy, _, _ = net.world(np.full(9, r), s, d)
            s2, d2, ok, _ = net.project(np.full(9, r), xy)
            self.assertTrue(ok.all())
            np.testing.assert_allclose(s2, s, atol=0.15)
            np.testing.assert_allclose(d2, d, atol=0.15)

    def test_crossing_and_merge_points(self):
        self.assertEqual(sc.crossing().conflicts(), [("ns", "ew", 50.0, 50.0, "cross")])
        net = sc.four_way()
        kinds = {int(k) for k in net.cf_kind[net.cf_valid]}
        self.assertEqual(kinds, {CROSS, MERGE})
        # routes from the same entry never get conflict points
        for a in net.routes:
            for b in net.routes:
                if a.lanes[0] == b.lanes[0]:
                    self.assertFalse(net.cf_valid[a.index, b.index].any())


class ConstantSpeedTests(unittest.TestCase):
    def test_moves_along_its_route_at_constant_speed(self):
        net = sc.four_way()
        for r in range(len(net.routes)):
            res = Simulation(net, ConstantSpeedModel(), [Arrival(0, "car", r, v=5.0, d=0.3)]).run(60)
            track = res.tracks()[0]
            np.testing.assert_allclose(track[:, 6], 5.0)                          # speed
            np.testing.assert_allclose(np.diff(track[:, 4]), 0.5, atol=1e-9)     # s advances 5 m/s * 0.1 s
            np.testing.assert_allclose(track[:, 5], 0.3)                          # offset never changes
            xy, _, _ = net.world(np.full(len(track), r), track[:, 4], track[:, 5])
            np.testing.assert_allclose(track[:, 2:4], xy, atol=1e-9)              # on its route

    def test_speed_and_offset_sources(self):
        net = sc.straight()
        model = ConstantSpeedModel({"motorcycle": 7.0}, default=4.0)
        res = Simulation(net, model, [Arrival(0, "motorcycle", 0, uid=0), Arrival(0, "car", 0, uid=1),
                                      Arrival(0, "car", 0, uid=2, v=3.0), Arrival(0, "car", 0, uid=3, v=3.0, v0=9.0)]).run(5)
        speed = {u: float(rows[0, 6]) for u, rows in res.tracks().items()}
        self.assertEqual(speed, {0: 7.0, 1: 4.0, 2: 3.0, 3: 9.0})
        right, left = net.edges(np.array(0), np.array(1.0))
        self.assertAlmostEqual(float(res.tracks()[0][0, 5]), float(right + left) / 2)

    def test_ignores_others_and_scores_contacts(self):
        # A fast car enters behind a slow one in the same line: it drives straight through it.
        net = sc.straight()
        res = Simulation(net, ConstantSpeedModel(), [Arrival(0, "car", 0, v=2.0, d=0.0, uid=0),
                                                     Arrival(3, "car", 0, v=8.0, d=0.0, uid=1)]).run(40)
        score = res.contact_score()
        self.assertEqual(score["contact_pairs"], 1)
        self.assertEqual(score["contact_events"], 1)
        self.assertGreater(score["contact_time_s"], 0.5)
        self.assertGreater(score["max_contact_depth_m"], 1.0)
        exits = {v["uid"]: v["t_exit"] for v in res.vehicles}
        self.assertIsNotNone(exits[1])
        self.assertIsNone(exits[0])

    def test_enters_on_time_even_when_occupied(self):
        net = sc.straight()
        res = Simulation(net, ConstantSpeedModel(), [Arrival(0, "car", 0, v=0.0), Arrival(1, "car", 0, v=0.0)]).run(5)
        self.assertEqual([v["t_spawn"] for v in res.vehicles], [0.0, 1.0])
        self.assertEqual(res.summary()["not_spawned"], 0)

    def test_deterministic(self):
        net = sc.four_way()
        arrivals = sc.poisson(net, 150, 60, seed=4)
        a = Simulation(net, ConstantSpeedModel(), [Arrival(x.t, x.kind, x.route, x.uid) for x in arrivals], seed=3).run(60)
        b = Simulation(net, ConstantSpeedModel(), [Arrival(x.t, x.kind, x.route, x.uid) for x in arrivals], seed=3).run(60)
        np.testing.assert_array_equal(a.samples, b.samples)

    def test_trajectory_export_round_trip(self):
        net = sc.straight()
        res = Simulation(net, ConstantSpeedModel(), [Arrival(0, "motorcycle", 0)]).run(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.trajectories.json"
            res.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "vehicle-trajectories")
            tracks, meta = load(path)
            self.assertTrue(meta["simulated"])
            self.assertEqual(len(tracks), 1)


class MeasureTests(unittest.TestCase):
    def test_kalman_recovers_speed(self):
        rng = np.random.default_rng(0)
        t = np.arange(0, 10, 0.04)
        truth = np.stack([5 * t, np.zeros_like(t)], axis=1)
        _, vel = kalman_smooth(t, truth + rng.normal(0, 0.3, truth.shape))
        self.assertAlmostEqual(float(np.median(vel[20:-20, 0])), 5.0, delta=0.15)

    def test_stitch_joins_a_broken_track(self):
        from twin.measure.tracks import Track
        t1, t2 = np.arange(0, 2, 0.1), np.arange(2.5, 4, 0.1)
        a = Track(1, "car", t1, np.stack([5 * t1, 0 * t1], 1), np.tile([5.0, 0.0], (len(t1), 1)),
                  raw_t=t1, raw_xy=np.stack([5 * t1, 0 * t1], 1), source_ids=[1])
        b = Track(2, "car", t2, np.stack([5 * t2, 0 * t2], 1), np.tile([5.0, 0.0], (len(t2), 1)),
                  raw_t=t2, raw_xy=np.stack([5 * t2, 0 * t2], 1), source_ids=[2])
        joined = stitch([a, b])
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0].source_ids, [1, 2])

    def test_wasserstein(self):
        self.assertAlmostEqual(wasserstein([0, 1, 2], [1, 2, 3]), 1.0)


class BehaviourInterfaceTests(unittest.TestCase):
    def test_engine_requires_a_behaviour_model(self):
        with self.assertRaises(TypeError):
            Simulation(sc.straight(), {"car": 5.0}, [])

    def test_model_decides_acceleration_and_drift(self):
        class Speeding(BehaviourModel):
            def sample_driver(self, kind, arrival, rng, right, left, width):
                return {"v": 0.0, "d": 0.0}

            def decide(self, view):
                n = len(view.slots)
                return Decision(acc=np.full(n, 1.0), vd=np.full(n, 0.1))

        res = Simulation(sc.straight(), Speeding(), [Arrival(0, "car", 0)]).run(10)
        track = res.tracks()[0]                                     # each sample is the state at its time t
        np.testing.assert_allclose(track[:, 6], 1.0 * track[:, 1], atol=1e-9)     # v = a t
        np.testing.assert_allclose(track[:, 5], 0.1 * track[:, 1], atol=1e-9)     # d = vd t
        np.testing.assert_allclose(track[:, 4] - track[0, 4], 0.5 * track[:, 1] ** 2, atol=1e-9)   # s = a t^2 / 2


if __name__ == "__main__":
    unittest.main()
