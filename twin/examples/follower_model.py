"""A small behaviour model kept outside twin/behaviour, as a template for adding one.

Each driver follows whatever is ahead of it in its line at a speed from which the time gap holds, and
otherwise drives at its own desired speed. It keeps its sideways offset and gives way to nobody, so it
is not a model of a junction: it shows what a model has to provide, in about forty lines.

Nothing registers it. The tools take it by path, and a fitted file remembers that path:

    python -m twin.calibrate.search --model twin.examples.follower_model:FollowerModel --layout L --video ... --out DIR
    python -m twin.calibrate.sensitivity --model twin.examples.follower_model:FollowerModel ...
    python -m twin.run --params DIR\\fitted.params.json --layout L --video V --out sim.trajectories.json

A model of your own can live in any importable module (a folder next to twin/ with an __init__.py, when
the tools are started from the project root). To make it a short name instead, put it in
twin/behaviour, decorate it with @register("name") and import it in twin/behaviour/__init__.py.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from twin.behaviour import BehaviourModel, Decision, Spec, View

RELAX_S = 0.5     # time to close the gap to the wanted speed


class FollowerModel(BehaviourModel):
    # name, default, lowest, highest, log scale, stage, meaning: all the search knows about a parameter.
    SPECS = (
        Spec("v0", 8.0, 3.0, 14.0, True, 1, "desired speed on a free road (m/s)"),
        Spec("v0_spread", 0.15, 0.02, 0.5, True, 1, "spread of desired speed between drivers (log-normal sigma)"),
        Spec("T", 1.0, 0.2, 2.5, True, 2, "time gap kept to the vehicle ahead (s)"),
        Spec("s0", 1.0, 0.2, 3.0, True, 2, "gap kept at standstill (m)"),
        Spec("a", 2.0, 0.5, 4.0, True, 2, "strongest acceleration (m/s2)"),
        Spec("b", 3.0, 1.0, 6.0, True, 2, "strongest braking (m/s2)"),
    )

    def resize(self, capacity, n_old):
        """Per-driver memory lives in arrays indexed by slot; the engine says when they must grow."""
        new = np.zeros(capacity)
        if n_old:
            new[:n_old] = self._v0[:n_old]
        self._v0 = new

    def sample_driver(self, kind, arrival, rng, right, left, width):
        p = self.params.values
        u = rng.random()        # one draw per driver whatever the parameters: runs with one seed stay comparable
        v0 = float(arrival.v0) if arrival.v0 is not None else p["v0"] * float(np.exp(p["v0_spread"] * ndtri(u)))
        return {"v": float(arrival.v) if arrival.v is not None else v0,
                "d": float(arrival.d) if arrival.d is not None else (right + left) / 2, "v0": v0}

    def enter(self, slot, driver):
        self._v0[slot] = driver["v0"]

    def decide(self, view: View) -> Decision:
        p = self.params.values
        # Pair rows: vehicle I looks at vehicle J. In my line: ahead, and overlapping me sideways.
        in_line = view.ok & view.ahead & (np.abs(view.D - view.d[view.I]) < view.lat_ext + view.half[view.I])
        gap = view.red_min(np.where(in_line, view.gap, np.inf))          # nearest one, per vehicle
        wanted = np.minimum(self._v0[view.slots], np.maximum(gap - p["s0"], 0.0) / p["T"])
        return Decision(acc=np.clip((wanted - view.v) / RELAX_S, -p["b"], p["a"]), vd=np.zeros(len(view.slots)))
