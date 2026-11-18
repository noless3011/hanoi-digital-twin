"""Checks of the priority model: it cannot deadlock, and its parameters do what they say."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from twin import scenarios as sc
from twin.behaviour import ParamSet, PriorityModel, make_model
from twin.behaviour.priority import default_params, specs_for
from twin.engine import Arrival, Simulation
from twin.measure.demand import apply_turning, turning_shifts


def run(net, arrivals, duration, params=None, **kw):
    return Simulation(net, PriorityModel(params or default_params()), arrivals, seed=1).run(duration, **kw)


class PriorityModelTests(unittest.TestCase):
    def test_no_deadlock_far_above_capacity(self):
        # A four-way junction without signals at 800 vehicles per hour per route: everybody gets out.
        net = sc.four_way()
        arrivals = sc.poisson(net, 800, 120, seed=3)
        res = run(net, arrivals, 120, until_empty=True)
        summary = res.summary()
        self.assertEqual(summary["exited"], len(arrivals))
        self.assertLess(summary["max_stopped_s"], 30)

    def test_the_most_senior_vehicle_is_never_held_up(self):
        net = sc.four_way()
        crowd = sc.poisson(net, 600, 60, seed=5)
        first = min(crowd, key=lambda a: a.t)
        alone = run(net, [Arrival(first.t, first.kind, first.route, uid=first.uid)], 120, until_empty=True)
        among = run(net, crowd, 60, until_empty=True)
        t_alone = alone.vehicles[0]["t_exit"]
        t_among = next(v for v in among.vehicles if v["uid"] == first.uid)["t_exit"]
        self.assertAlmostEqual(t_alone, t_among, delta=0.11)

    def test_seniority_stays_a_strict_order(self):
        net = sc.four_way()
        model = PriorityModel(default_params())
        sim = Simulation(net, model, sc.poisson(net, 600, 60, seed=2), seed=1)
        for _ in range(400):
            sim.step()
        keys = model._key[np.flatnonzero(sim.active)]
        self.assertGreater(len(keys), 10)
        self.assertEqual(len(np.unique(keys)), len(keys))

    def test_follows_without_contact(self):
        res = run(sc.straight(half_width=0.6), [Arrival(0, "car", 0, v=3.0, v0=3.0, d=0.0),
                                                Arrival(4, "car", 0, v=8.0, v0=9.0, d=0.0)], 40)
        self.assertEqual(res.contact_score()["contact_pairs"], 0)
        self.assertAlmostEqual(float(res.tracks()[1][-1, 6]), 3.0, delta=0.3)

    def test_a_faster_senior_vehicle_behind_follows_the_junior_one_ahead(self):
        # The junior vehicle is put ahead by entering on a shorter route that joins the same exit road.
        net = sc.four_way()
        names = sc.route_names(net)
        via = {v: k for k, v in names.items()}
        res = run(net, [Arrival(0, "car", via["NS"], v=9.0, v0=9.0, d=0.0),       # senior, long way round
                        Arrival(1, "car", via["ES"], v=3.0, v0=3.0, d=0.0)], 90)  # junior, joins ahead of it
        exits = {v["uid"]: v["t_exit"] for v in res.vehicles}
        self.assertTrue(all(t is not None for t in exits.values()))
        self.assertEqual(res.contact_score()["contact_pairs"], 0)

    def test_crossing_pairs_get_through(self):
        for kinds in (("motorcycle", "motorcycle"), ("car", "motorcycle"), ("car", "car")):
            res = run(sc.crossing(), [Arrival(0, kinds[0], 0, d=0.0), Arrival(0, kinds[1], 1, d=0.0)], 60)
            summary = res.summary()
            self.assertEqual((summary["exited"], summary["contact_pairs"]), (2, 0), kinds)

    def test_speed_follows_the_route_profile(self):
        params = ParamSet({"v_r0_k0": 4.0, "v_r0_k1": 10.0, "v_r0_k2": 10.0, "v_r0_k3": 4.0, "keep": 0.0, "spread": 0.02},
                          specs=specs_for([0]), model="priority")
        res = run(sc.straight(length=300.0), [Arrival(0, "motorcycle", 0, d=0.0)], 80, params)
        track = res.tracks()[0]
        s, v = track[:, 4], track[:, 6]
        middle = (s > 120) & (s < 180)
        self.assertGreater(middle.sum(), 10)
        self.assertGreater(float(v[middle].min()), 9.0)
        self.assertLess(float(v[s > 285].mean()), 6.0)

    def test_parameter_files_remember_their_model(self):
        params = default_params([0, 2], turns=2).with_values({"v_r2_k1": 5.5, "turn_2": -1.0})
        with tempfile.TemporaryDirectory() as tmp:
            params.save(Path(tmp) / "p.json")
            loaded = ParamSet.load(Path(tmp) / "p.json")
        self.assertEqual(loaded.values, params.values)
        self.assertEqual(loaded.names, params.names)
        self.assertIsInstance(make_model(loaded), PriorityModel)
        x = loaded.to_unit(loaded.names)
        self.assertTrue(all(0 <= u <= 1 for u in x))


class TurningTests(unittest.TestCase):
    def setUp(self):
        self.arrivals = [Arrival(float(k), "motorcycle", 0, uid=k, d=0.0, v=5.0) for k in range(40)]
        self.turning = {"routes": [0, 1, 2], "prior": [0.5, 0.3, 0.2],
                        "options": {k: {"u": (k + 0.5) / 20, "routes": {0: (float(k), 0.0), 1: (k + 0.5, 1.0)}}
                                    for k in range(20)}}

    def routes(self, shifts):
        return [a.route for a in sorted(apply_turning(self.arrivals, self.turning, shifts), key=lambda a: a.uid)]

    def test_shares_move_the_ambiguous_arrivals_only(self):
        base = self.routes([0.0, 0.0])
        self.assertEqual(base[20:], [0] * 20)                              # one possible route: untouched
        self.assertAlmostEqual(base[:20].count(1) / 20, 0.3 / 0.8, delta=0.06)   # prior, among its two options
        self.assertEqual(self.routes([8.0, 0.0])[:20], [1] * 20)
        self.assertEqual(self.routes([-8.0, 0.0])[:20], [0] * 20)

    def test_a_vehicle_changes_route_at_most_once_as_a_share_grows(self):
        seen = np.array([self.routes([x, 0.0])[:20] for x in np.linspace(-4, 4, 17)])
        self.assertTrue((np.diff(seen, axis=0) >= 0).all())

    def test_entry_time_and_offset_come_with_the_route(self):
        moved = {a.uid: a for a in apply_turning(self.arrivals, self.turning, [8.0, 0.0])}
        self.assertEqual((moved[3].route, moved[3].t, moved[3].d), (1, 3.5, 1.0))
        self.assertEqual(turning_shifts({"turn_1": 0.4, "turn_2": -1.0}, self.turning), [0.4, -1.0])


if __name__ == "__main__":
    unittest.main()
