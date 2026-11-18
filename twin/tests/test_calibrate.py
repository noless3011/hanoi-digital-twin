"""End-to-end checks of the calibration tools (twin.calibrate.search and .sensitivity) on a small clip.

The "video" is a simulation on the Bưởi layout: its file gives the arrivals, and the search's --truth
option replaces what the camera saw with a simulation under known parameters. Real clips are not in
git, so this is what can run anywhere. The model is the plug-in example, named by path, which also
checks that worker processes can load a model that is not registered. About 30 s.
"""
import io
import json
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from twin import run
from twin.behaviour import ParamSet, default_params, make_model
from twin.calibrate import search, sensitivity
from twin.examples.follower_model import FollowerModel

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "simulation/Assets/StreamingAssets/Traffic/buoi-hht.camera-lanes.json"
MODEL = "twin.examples.follower_model:FollowerModel"
TRUE_V0, START_V0 = 4.5, 8.0


def quiet(main, argv):
    with redirect_stdout(io.StringIO()) as out:
        code = main([str(a) for a in argv])
    return code, out.getvalue()


@unittest.skipUnless(LAYOUT.exists(), "needs the Bưởi camera lane layout")
class CalibrationToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.clip = cls.tmp / "clip.trajectories.json"
        quiet(run.main, ["--layout", LAYOUT, "--rate", 400, "--minutes", 1.5, "--seed", 3, "--model", "constant",
                         "--out", cls.clip])
        cls.truth = cls.tmp / "truth.params.json"
        default_params(MODEL).with_values({"v0": TRUE_V0}).save(cls.truth)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_search_recovers_a_known_parameter(self):
        out = self.tmp / "search"
        code, log = quiet(search.main, ["--layout", LAYOUT, "--video", self.clip, "--truth", self.truth,
                                        "--model", MODEL, "--names", "v0", "--generations", 8, "--popsize", 4,
                                        "--check-seeds", 11, "--workers", 2, "--out", out])
        self.assertEqual(code, 0, log)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["model"], MODEL)
        row = report["parameters"]["v0"]
        self.assertEqual(row["start"], START_V0)
        self.assertLess(row["error_share_of_range"], 0.1, row)         # within 10% of [3, 14] m/s
        self.assertLess(report["after"]["fit"]["objective"], report["before"]["fit"]["objective"])
        fitted = ParamSet.load(out / "fitted.params.json")
        self.assertEqual(fitted.model, MODEL)
        self.assertIsInstance(make_model(fitted), FollowerModel)
        self.assertEqual(set(fitted.names), {s.name for s in FollowerModel.SPECS})   # untouched ones kept
        # Each generation ran its own seeds; the answer was chosen on seeds the search did not use.
        self.assertTrue(report["seeds"]["rotate"])
        self.assertIn(report["selection"]["taken"], report["selection"]["select_objective"])

    def test_generation_seeds(self):
        self.assertEqual(search.generation_seeds([1, 2], 0), [1, 2])
        self.assertEqual(search.generation_seeds([1, 2], 3), [3001, 3002])
        self.assertEqual(search.generation_seeds([1, 2], 3, rotate=False), [1, 2])

    def test_sensitivity_reports_every_parameter_and_resumes(self):
        out = self.tmp / "sensitivity"
        argv = ["--layout", LAYOUT, "--video", self.clip, "--model", MODEL, "--trajectories", 2, "--workers", 2,
                "--out", out]
        code, log = quiet(sensitivity.main, argv)
        self.assertEqual(code, 0, log)
        report = json.loads((out / "sensitivity.json").read_text(encoding="utf-8"))
        names = [s.name for s in FollowerModel.SPECS]
        self.assertEqual(report["names"], names)
        self.assertEqual(set(report["effects"]), set(names))
        for name in names:
            self.assertGreaterEqual(report["effects"][name]["score"]["effect"], 0.0)
        self.assertIn("score", report["noise_floor"])
        # 2 trajectories x (6 steps + start + noise check) x 1 clip, each saved as it finishes.
        lines = (out / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2 * (len(names) + 2))
        code, log = quiet(sensitivity.main, argv)                      # a rerun simulates nothing new
        self.assertEqual(code, 0, log)
        self.assertIn(f"resuming: {len(lines)} simulations already done", log)
        self.assertEqual(len((out / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()), len(lines))

    def test_unknown_model_is_a_usage_error(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                unittest.mock.patch("sys.stderr", io.StringIO()):
            search.main([str(a) for a in ["--layout", LAYOUT, "--video", self.clip, "--model", "nonsense",
                                          "--out", self.tmp / "x"]])


if __name__ == "__main__":
    unittest.main()
