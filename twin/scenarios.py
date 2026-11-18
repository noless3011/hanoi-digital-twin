"""Synthetic roads and demand for the structural checks.

All roads are built from the same lanes and routes as a real layout, so the checks exercise exactly the
code that runs on video scenes.
"""
from __future__ import annotations

import numpy as np

from .engine import Arrival
from .network import Network, lane_from_polyline
from .signals import SignalPlan

DEFAULT_MIX = {"motorcycle": 0.8, "car": 0.2}


def _line(a, b, step=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    count = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1) + 1
    return a + (b - a) * np.linspace(0, 1, count)[:, None]


def _bezier(a, c, b, step=0.5):
    a, c, b = (np.asarray(x, float) for x in (a, c, b))
    approx = np.linalg.norm(c - a) + np.linalg.norm(b - c)
    t = np.linspace(0, 1, max(int(np.ceil(approx / step)), 2) + 1)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def straight(length=200.0, half_width=3.5):
    lane = lane_from_polyline("road", "A", "B", _line((0, 0), (length, 0)), half_width)
    return Network([lane], routes=[("road",)], name="straight")


def head_on(length=120.0, half_width=2.5):
    """One two-way road of shared width: both directions use the same centreline and edges."""
    east = lane_from_polyline("east", "W", "E", _line((0, 0), (length, 0)), half_width)
    west = lane_from_polyline("west", "E", "W", _line((length, 0), (0, 0)), half_width)
    return Network([east, west], routes=[("east",), ("west",)], name="head-on")


def crossing(arm=50.0, half_width=3.0):
    ns = lane_from_polyline("ns", "S", "N", _line((0, -arm), (0, arm)), half_width)
    ew = lane_from_polyline("ew", "W", "E", _line((-arm, 0), (arm, 0)), half_width)
    return Network([ns, ew], routes=[("ns",), ("ew",)], name="crossing")


ARMS = {"N": (0., 1.), "E": (1., 0.), "S": (0., -1.), "W": (-1., 0.)}


def four_way(arm=80.0, box=10.0, carriageway=5.0):
    """Four two-way arms meeting in a box. Each direction has its own carriageway (one wide lane);
    every entry can go straight, left or right (12 routes)."""
    off, half = carriageway / 2, carriageway / 2
    lanes, ends = [], {}
    for name, (ux, uy) in ARMS.items():
        u = np.array([ux, uy])
        right_in, right_out = np.array([-uy, ux]), np.array([uy, -ux])
        a_in, b_in = u * (box + arm) + off * right_in, u * box + off * right_in
        a_out, b_out = u * box + off * right_out, u * (box + arm) + off * right_out
        lanes.append(lane_from_polyline(f"in_{name}", f"{name}_far_in", f"{name}_box_in", _line(a_in, b_in), half))
        lanes.append(lane_from_polyline(f"out_{name}", f"{name}_box_out", f"{name}_far_out", _line(a_out, b_out), half))
        ends[name] = (b_in, -u, a_out, u)
    routes = []
    for src, (p_in, t_in, _, _) in ends.items():
        for dst, (_, _, p_out, t_out) in ends.items():
            if src == dst:
                continue
            # control point: where the entry and exit directions meet (straight: the midpoint)
            m = np.array([[t_in[0], -t_out[0]], [t_in[1], -t_out[1]]])
            if abs(np.linalg.det(m)) < 1e-6:
                c = (p_in + p_out) / 2
            else:
                k = np.linalg.solve(m, p_out - p_in)[0]
                c = p_in + t_in * k
            lanes.append(lane_from_polyline(f"{src}{dst}", f"{src}_box_in", f"{dst}_box_out", _bezier(p_in, c, p_out), half))
            routes.append((f"in_{src}", f"{src}{dst}", f"out_{dst}"))
    return Network(lanes, routes=routes, name="four-way")


def route_names(network):
    """Short names for four-way routes: 'NS' is from north to south."""
    return {r.index: r.lanes[1] if len(r.lanes) == 3 else r.id for r in network.routes}


def four_way_signals(network, cycle=60.0, yellow=3.0, arm=80.0):
    groups = [{"name": "NS", "routes": {}, "green": [0.0, cycle / 2 - yellow]},
              {"name": "EW", "routes": {}, "green": [cycle / 2, cycle - yellow]}]
    for r in network.routes:
        src = r.lanes[0][-1]
        groups[0 if src in "NS" else 1]["routes"][r.id] = r.lane_start[r.lanes[1]] - 0.5
    return SignalPlan(cycle, groups, yellow)


def poisson(network, rate_per_hour, duration, *, mix=None, seed=0, routes=None):
    """Random arrivals: rate_per_hour per route (a number, or {route index: rate}), kinds drawn from mix."""
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    kinds, shares = list(mix), np.array(list(mix.values()), float)
    shares /= shares.sum()
    routes = range(len(network.routes)) if routes is None else routes
    arrivals = []
    for r in routes:
        rate = rate_per_hour[r] if isinstance(rate_per_hour, dict) else rate_per_hour
        if rate <= 0:
            continue
        t = rng.exponential(3600 / rate)
        while t < duration:
            arrivals.append(Arrival(float(t), kinds[rng.choice(len(kinds), p=shares)], int(r)))
            t += rng.exponential(3600 / rate)
    arrivals.sort()
    for k, a in enumerate(arrivals):
        a.uid = k
    return arrivals
