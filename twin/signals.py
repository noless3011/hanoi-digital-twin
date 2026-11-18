"""Fixed-time traffic signals.

A plan has a cycle length and groups. Each group lists the routes it controls, the station of the stop
line on each of them, and when in the cycle it is green. Yellow follows green for `yellow` seconds;
the rest of the cycle is red.

    {"cycle": 60, "yellow": 3, "offset": 0,
     "groups": [{"name": "NS", "routes": {"N>S": 95.0, "S>N": 95.0}, "green": [0, 27]}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GREEN, YELLOW, RED = 0, 1, 2
NAMES = ("green", "yellow", "red")


class SignalPlan:
    def __init__(self, cycle, groups, yellow=3.0, offset=0.0):
        self.cycle, self.yellow, self.offset = float(cycle), float(yellow), float(offset)
        self.groups = groups

    @classmethod
    def from_json(cls, data):
        return cls(data["cycle"], data["groups"], data.get("yellow", 3.0), data.get("offset", 0.0))

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self):
        return {"cycle": self.cycle, "yellow": self.yellow, "offset": self.offset, "groups": self.groups}

    def state(self, group, t):
        start, end = self.groups[group]["green"]
        c = (t - self.offset) % self.cycle
        into = (c - start) % self.cycle
        green = (end - start) % self.cycle
        if into < green:
            return GREEN
        if into < green + self.yellow:
            return YELLOW
        return RED

    def bind(self, network):
        """Per-route arrays: the controlling group (-1 for none) and the stop-line station."""
        group = np.full(len(network.routes), -1)
        stop = np.full(len(network.routes), np.inf)
        for g, spec in enumerate(self.groups):
            for route_id, station in spec["routes"].items():
                index = network.route_ids[route_id]
                group[index], stop[index] = g, float(station)
        return group, stop

    def states(self, t):
        return np.array([self.state(g, t) for g in range(len(self.groups))], dtype=np.int8)
