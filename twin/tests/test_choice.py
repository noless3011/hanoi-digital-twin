"""Checks of the choice model, its parameter set, and the image-space score."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from twin import scenarios as sc
from twin.behaviour import ChoiceModel, ConstantSpeedModel, ParamSet
from twin.behaviour.params import NAMES
from twin.engine import Arrival, Simulation
from twin.measure import image, image_score


def run(net, arrivals, duration, params=None):
    return Simulation(net, ChoiceModel(params or ParamSet()), arrivals, seed=1).run(duration)


def synthetic_layout(width=384, height=216):
    """A camera 12 m up looking down a straight road along +x (Unity x, z on the ground)."""
    eye, target, up = np.array([-10.0, 12.0, 0.0]), np.array([40.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(up, f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = r, u, -f          # camera looks down -z (OpenGL)
    view[:3, 3] = -view[:3, :3] @ eye
    near, far, t = 0.1, 500.0, np.tan(np.radians(50) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1] = 1 / (t * width / height), 1 / t
    proj[2, 2], proj[2, 3], proj[3, 2] = -(far + near) / (far - near), -2 * far * near / (far - near), -1
    row = lambda x, half: {"center": [x, 0, 0], "left": [x, 0, half], "right": [x, 0, -half]}
    return {"image": {"width": width, "height": height}, "ground_y": 0.0,
            "camera": {"world_to_clip": (proj @ view).ravel().tolist(), "near_m": near},
            "lanes": [{"id": "road0000", "name": "road", "sections": [row(0, 3.5), row(30, 3.5), row(60, 3.5)]}]}


class ParamSetTests(unittest.TestCase):
    def test_unit_round_trip_and_bounds(self):
        p = ParamSet({"v0": 6.0, "tau": 2.5})
        x = p.to_unit(NAMES)
        self.assertTrue(all(0 <= u <= 1 for u in x))
        q = ParamSet().from_unit(NAMES, x)
        for name in NAMES:
            self.assertAlmostEqual(q.values[name], p.values[name], places=9)
        self.assertAlmostEqual(ParamSet().from_unit(["v0"], [2.0]).values["v0"], 14.0)   # clipped to the bound

    def test_kinds_and_files(self):
        p = ParamSet({"T": 1.0})
        self.assertAlmostEqual(p.of("car")["T"], 1.4)
        with tempfile.TemporaryDirectory() as tmp:
            p.save(Path(tmp) / "p.json")
            self.assertEqual(ParamSet.load(Path(tmp) / "p.json").values, p.values)
        with self.assertRaises(ValueError):
            ParamSet({"nonsense": 1.0})


class ChoiceModelTests(unittest.TestCase):
    def test_alone_it_reaches_and_holds_its_desired_speed_and_position(self):
        res = run(sc.straight(), [Arrival(0, "motorcycle", 0, v=2.0, v0=8.0, d=0.5)], 30)
        track = res.tracks()[0]
        self.assertAlmostEqual(float(track[-1, 6]), 8.0, delta=0.05)
        np.testing.assert_allclose(track[:, 5], 0.5, atol=1e-9)

    def test_follows_without_contact_when_it_cannot_pass(self):
        res = run(sc.straight(half_width=0.6), [Arrival(0, "car", 0, v=3.0, v0=3.0, d=0.0),
                                                Arrival(4, "car", 0, v=8.0, v0=9.0, d=0.0)], 40)
        self.assertEqual(res.contact_score()["contact_pairs"], 0)
        tracks = res.tracks()
        self.assertAlmostEqual(float(tracks[1][-1, 6]), 3.0, delta=0.4)        # settled at the leader's speed
        self.assertGreater(float(tracks[0][-1, 4] - tracks[1][-1, 4]) - 4.4, 1.0)   # with a gap

    def test_passes_a_slow_vehicle_on_a_wide_road(self):
        res = run(sc.straight(), [Arrival(0, "car", 0, v=3.0, v0=3.0, d=0.0),
                                  Arrival(4, "motorcycle", 0, v=8.0, v0=9.0, d=0.0)], 40)
        self.assertEqual(res.contact_score()["contact_pairs"], 0)
        exits = {v["uid"]: v["t_exit"] for v in res.vehicles}
        self.assertIsNotNone(exits[1])       # the motorcycle got past and out
        self.assertIsNone(exits[0])          # while the car is still on the road

    def test_stops_behind_a_crawling_vehicle(self):
        res = run(sc.straight(half_width=0.6), [Arrival(0, "car", 0, v=0.3, v0=0.3, d=0.0),
                                                Arrival(60, "motorcycle", 0, v=9.0, v0=9.0, d=0.0)], 100)
        self.assertEqual(res.contact_score()["contact_pairs"], 0)

    def test_pairs_get_past_each_other(self):
        for net, routes in ((sc.head_on(), (0, 1)), (sc.crossing(), (0, 1))):
            for kinds in (("motorcycle", "motorcycle"), ("car", "motorcycle"), ("car", "car")):
                res = run(net, [Arrival(0, kinds[0], routes[0], d=0.0), Arrival(0, kinds[1], routes[1], d=0.0)], 60)
                summary = res.summary()
                self.assertEqual(summary["contact_pairs"], 0, (net.name, kinds))
                self.assertEqual(summary["exited"], 2, (net.name, kinds))

    def test_same_seed_same_run_and_draws_do_not_depend_on_parameters(self):
        net = sc.four_way()
        arrivals = lambda: sc.poisson(net, 120, 60, seed=4)
        a, b = run(net, arrivals(), 60), run(net, arrivals(), 60)
        np.testing.assert_array_equal(a.samples, b.samples)
        # Changing one parameter leaves every driver's random share of the others unchanged.
        m1, m2 = ChoiceModel(ParamSet()), ChoiceModel(ParamSet({"T": 1.5}))
        rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
        for _ in range(5):
            d1 = m1.sample_driver("motorcycle", Arrival(0, "motorcycle", 0), rng1, -3, 3, 0.75)
            d2 = m2.sample_driver("motorcycle", Arrival(0, "motorcycle", 0), rng2, -3, 3, 0.75)
            self.assertEqual((d1["v0"], d1["caution"], d1["d"]), (d2["v0"], d2["caution"], d2["d"]))

    def test_a_driver_seen_on_video_keeps_its_own_speed(self):
        model = ChoiceModel(ParamSet({"v0_seen": 1.2}))
        rng = np.random.default_rng(3)
        seen = model.sample_driver("car", Arrival(0, "car", 0, v_seen=5.0), rng, -3, 3, 1.8)
        self.assertAlmostEqual(seen["v0"], 6.0)                     # v0_seen x seen speed; no kind ratio
        given = model.sample_driver("car", Arrival(0, "car", 0, v0=9.0, v_seen=5.0), rng, -3, 3, 1.8)
        self.assertEqual(given["v0"], 9.0)                          # a given desired speed comes first

    def test_far_fewer_contacts_than_constant_speed(self):
        net = sc.four_way()
        arrivals = lambda: sc.poisson(net, 150, 180, seed=3)
        blind = Simulation(net, ConstantSpeedModel(), arrivals(), seed=1).run(180).contact_score()["contact_time_s"]
        seeing = run(net, arrivals(), 180).contact_score()["contact_time_s"]
        self.assertLess(seeing, 0.2 * blind)


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.camera = image.Camera(synthetic_layout())
        self.gates = image.gates(self.camera)
        self.net = sc.straight(length=60.0)

    def boxes(self, model, seed=1, rate=900):
        res = Simulation(self.net, model, sc.poisson(self.net, rate, 120, seed=2), seed=seed).run(120)
        return image.from_result(res, self.camera)

    def test_projection_and_scale(self):
        pts, front = self.camera.project(np.array([[40.0, 0.0, 0.0], [-50.0, 0.0, 0.0]]))
        np.testing.assert_allclose(pts[0], [0.5, 0.5], atol=1e-6)     # the point the camera looks at
        self.assertTrue(front[0] and not front[1])
        near, far = self.camera.scale(192, 200), self.camera.scale(192, 90)
        self.assertGreater(near, far)                                  # more pixels per metre close by
        self.assertEqual(float(self.camera.scale(192, 2)), 0.0)        # sky
        # One metre along the road at the looked-at point covers about `scale` pixels (to within the
        # foreshortening that makes along-view and across-view pixels differ).
        a, _ = self.camera.project(np.array([[40.0, 0.0, -0.5], [40.0, 0.0, 0.5]]))
        across = abs(a[1, 0] - a[0, 0]) * 384
        self.assertGreater(across, float(self.camera.scale(192, 108)))

    def test_gate_counts_every_vehicle_once(self):
        self.assertEqual(len(self.gates), 1)
        res = Simulation(self.net, ConstantSpeedModel(), [Arrival(t, "car", 0, v=6.0, d=0.0) for t in (0, 5, 10)]).run(60)
        tracks = image.from_result(res, self.camera)
        times = image_score.crossings(tracks, self.gates)[self.gates[0].name]
        self.assertEqual(len(times), 3)
        np.testing.assert_allclose(np.diff(times), 5.0, atol=0.11)

    def test_velocity_is_about_metres_per_second(self):
        res = Simulation(self.net, ConstantSpeedModel(), [Arrival(0, "car", 0, v=6.0, d=0.0)]).run(20)
        tracks = image.from_result(res, self.camera)
        _, u = image_score.velocities(tracks)
        speed = np.linalg.norm(u, axis=1)
        self.assertGreater(len(speed), 20)
        self.assertTrue(2.0 < float(np.median(speed)) < 12.0)

    def test_same_run_scores_zero_and_a_slower_one_does_not(self):
        fast = image_score.features(self.boxes(ConstantSpeedModel({"motorcycle": 8.0, "car": 8.0})), self.gates)
        slow = image_score.features(self.boxes(ConstantSpeedModel({"motorcycle": 4.0, "car": 4.0})), self.gates)
        self.assertLess(image_score.score(fast, fast)["total"], 1e-9)
        report = image_score.score(fast, slow)
        self.assertGreater(report["parts"]["motion"], 0.2)
        self.assertGreater(report["total"], 0.1)

    def test_score_orders_parameters(self):
        # A simulation under other parameters is further from the "video" than one under the same
        # parameters and another seed.
        truth = ParamSet({"v0": 6.0})
        video = image_score.features(self.boxes(ChoiceModel(truth), seed=5), self.gates)
        same = image_score.score(video, image_score.features(self.boxes(ChoiceModel(truth), seed=6), self.gates))
        other = image_score.score(video, image_score.features(self.boxes(ChoiceModel(ParamSet({"v0": 11.0})), seed=6), self.gates))
        self.assertLess(same["total"], other["total"])


if __name__ == "__main__":
    unittest.main()
