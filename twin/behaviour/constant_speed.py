"""Constant-speed model: the simplest driver. Each vehicle follows its route at one speed, keeps the
sideways offset it entered at (no lane changes), and ignores everyone else (contacts are only scored).

Speed of a driver, first that is known:
  1. arrival.v0  a desired speed given for this vehicle
  2. arrival.v   its speed at entry (from video: the speed it was first seen at)
  3. speeds[kind] the model's speed for its kind
  4. DEFAULT_SPEED

Offset at entry: arrival.d when given (from video: where it was first seen), otherwise the middle of the
road. Both are kept for the whole route.

    model = ConstantSpeedModel({"motorcycle": 6.5, "car": 6.0})
    result = Simulation(network, model, arrivals).run(180)
"""
from __future__ import annotations

import numpy as np

from .base import BehaviourModel, Decision, View

DEFAULT_SPEED = 6.0   # m/s


class ConstantSpeedModel(BehaviourModel):
    def __init__(self, speeds=None, default=DEFAULT_SPEED):
        self.speeds = dict(speeds or {})
        self.default = float(default)

    def speed_of(self, kind, arrival):
        for value in (arrival.v0, arrival.v):
            if value is not None:
                return float(value)
        return float(self.speeds.get(kind, self.default))

    def sample_driver(self, kind, arrival, rng, right, left, width):
        d = arrival.d if arrival.d is not None else (right + left) / 2
        return {"v": self.speed_of(kind, arrival), "d": float(d)}

    def decide(self, view: View) -> Decision:
        n = len(view.slots)
        return Decision(acc=np.zeros(n), vd=np.zeros(n))
