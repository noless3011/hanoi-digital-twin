"""Priority model: a simple driver that cannot deadlock, with parameters a video can pin down.

Every vehicle has a seniority (the order it entered in). A driver reacts only to vehicles senior to it:
it follows them, and waits for them to cross, at a speed from which it can still stop (Gipps' safe
speed). It ignores everyone junior. The most senior vehicle on the road therefore reacts to nobody and
always drives on; when it has left, the next one is the most senior; so no ring of drivers waiting for
each other can form, whatever the parameters. The price: where two vehicles would lock, the senior one
drives through the junior one, and the engine records the contact.

A senior vehicle coming the other way is not waited for (stopping does not help): the junior driver
moves aside, away from its line. Three rules keep vehicles apart without breaking that order, because each changes only what the
junior driver does:
  separate   a junior vehicle whose body overlaps (or almost touches) a senior one brakes if the senior
             one is ahead of it and in line with it, and steers away from it in every case;
  enter      an arriving vehicle takes the free place across the road nearest to where it was seen;
  look aside a driver drifting back to its own position stops drifting when anyone is beside it on that side.

One exchange keeps following sensible: a junior vehicle directly ahead of a senior one, going the same
way, swaps seniority with it (the one in front leads). Seniority stays a strict order.

What a driver wants is a speed that depends on where it is: each busy route has a speed profile (KNOTS
values along its length, straight lines between), times the driver's own factor. That factor blends the
speed the vehicle was first seen at with a random draw (`keep`: 1 all observed, 0 all drawn). Sideways,
a driver holds the share of the road's width it entered at.

Parameters (specs_for builds the list for a network's busy routes):

  v_r<route>_k<knot>  speed on that route at that point of its length (m/s)
  v_free              speed on routes without a profile (m/s)
  keep, spread        how much of its observed entry speed a driver keeps; spread of the random part
  large               speed of cars, vans, trucks and buses relative to two-wheelers
  T, s0, clat         time gap and standstill gap kept to a senior vehicle; sideways clearance within
                      which a senior vehicle is in the way
  turn_<k>            demand, not behaviour: shift of the main entry's turning shares (measure/demand.py)
"""
from __future__ import annotations

import re

import numpy as np
from scipy.special import ndtri

from .base import BehaviourModel, Decision, View
from ..vehicles import DIMENSIONS
from .params import ParamSet, Spec, turning_specs
from .registry import register

KNOTS = 4
ACCELERATION = 2.0      # m/s2
BRAKING = 3.0           # braking assumed in the safe speed (m/s2)
HARD_BRAKING = 6.0      # the most a driver brakes (m/s2)
RELAX_S = 0.5           # time to close the gap to the wanted speed
LOOK_S = 2.0            # how far ahead a senior vehicle's sideways movement is predicted
SWAP_RANGE_M = 20.0     # a junior vehicle this close ahead, in line, takes over seniority
SIDE_SPEED = 0.8        # fastest sideways drift (m/s)
DODGE_RANGE_M = 25.0    # an oncoming senior vehicle this close in my line is moved aside for
TOUCH_MARGIN_M = 0.15   # bodies closer than this (along and across) count as touching, for separating
IN_LINE_OVERLAP_M = 0.3  # sideways overlap from which a touching vehicle ahead is braked for (m)
SEPARATE_SPEED = 1.2    # sideways speed when moving off a vehicle being touched (m/s)
ENTRY_MARGIN_M = 0.2    # sideways room kept to vehicles at the entry when choosing where to enter
ENTRY_REACH_M = 3.0     # vehicles whose rear is within this of the entering vehicle's front block a place
BESIDE_MARGIN_M = 0.3   # a vehicle this close sideways (and overlapping along) stops a drift towards it
TWO_WHEEL = ("motorcycle", "bicycle")
FACTOR_RANGE = (0.3, 2.5)
BUSY_ROUTE_SHARE = 0.05  # routes with at least this share of a scene's arrivals get a speed profile


def specs_for(routes=(), turns=0, knots=KNOTS):
    """The parameter list: speed profiles for `routes` (route indices), `turns` turning-share shifts."""
    specs = [Spec("v_free", 7.0, 2.5, 13.0, True, 1, "speed on routes without a profile (m/s)")]
    for r in routes:
        for k in range(knots):
            specs.append(Spec(f"v_r{r}_k{k}", 7.0, 1.5, 13.0, True, 1,
                              f"speed on route {r} at {k}/{knots - 1} of its length (m/s)"))
    specs += [
        Spec("keep", 0.5, 0.0, 1.0, False, 1, "share of its observed entry speed a driver keeps"),
        Spec("spread", 0.15, 0.02, 0.5, True, 1, "spread of the random part of a driver's speed (log-normal sigma)"),
        Spec("large", 0.9, 0.5, 1.2, False, 1, "speed of cars, vans, trucks, buses relative to two-wheelers"),
        Spec("T", 1.0, 0.2, 2.5, True, 2, "time gap kept to a senior vehicle (s)"),
        Spec("s0", 1.0, 0.2, 3.0, True, 2, "gap kept to a senior vehicle at standstill (m)"),
        Spec("clat", 0.3, 0.05, 1.5, True, 2, "sideways clearance within which a senior vehicle is in the way (m)"),
    ]
    return specs + turning_specs(turns)


def default_params(routes=(), turns=0):
    return ParamSet(specs=specs_for(routes, turns), model="priority")


def busy_routes(route_arrivals):
    counts = np.asarray(route_arrivals, float)
    return [int(r) for r in np.flatnonzero(counts >= BUSY_ROUTE_SHARE * counts.sum())] if counts.sum() > 0 else []


@register("priority")
class PriorityModel(BehaviourModel):
    @classmethod
    def default_params(cls, scene=None):
        """A speed profile for each busy route of the scene, and its turning shares."""
        if scene is None:
            return default_params()
        return default_params(busy_routes(scene.route_arrivals) if scene.route_arrivals is not None else (), scene.turns)

    def __init__(self, params=None):
        super().__init__(params)
        self._entered = 0

    # ------------------------------------------------------------ drivers

    PER_SLOT = ("factor", "pref", "key")

    def bind(self, network, capacity):
        values = self.params.values
        knots = max([int(m.group(2)) for m in (re.fullmatch(r"v_r(\d+)_k(\d+)", n) for n in values) if m] + [0]) + 1
        self._profile = np.full((len(network.routes), max(knots, 2)), values["v_free"])
        for name, value in values.items():
            m = re.fullmatch(r"v_r(\d+)_k(\d+)", name)
            if m and int(m.group(1)) < len(network.routes):
                self._profile[int(m.group(1)), int(m.group(2))] = value
        self._entering = []
        self._seen = None
        super().bind(network, capacity)

    def resize(self, capacity, n_old):
        for name in self.PER_SLOT:
            new = np.zeros(capacity)
            if n_old:
                new[:n_old] = getattr(self, "_" + name)[:n_old]
            setattr(self, "_" + name, new)

    def wanted_speed(self, route, s):
        """The route's profile at station s (before the driver's own factor)."""
        last = self._profile.shape[1] - 1
        x = np.clip(s / self.net.length[route], 0, 1) * last
        k = np.minimum(x.astype(np.int64), last - 1)
        f = x - k
        return self._profile[route, k] * (1 - f) + self._profile[route, k + 1] * f

    def _free_offset(self, route, wanted, length, width, right, left):
        """The offset nearest `wanted` where an entering body of this size does not meet anyone: the
        vehicles seen last step, and those that entered since."""
        lo, hi = right + width / 2, left - width / 2
        if hi <= lo:
            return wanted
        blocked = []
        seen = getattr(self, "_seen", None)
        if seen is not None and len(seen[0]):
            xy, L, W = seen
            st, dd, ok, _ = self.net.project(np.full(len(xy), route), xy)
            near = ok & (st - L / 2 < length + ENTRY_REACH_M)
            blocked += [(float(x), float(w)) for x, w in zip(dd[near], W[near])]
        blocked += [(d, w) for r, d, w in self._entering if r == route]
        if not blocked:
            return wanted
        grid = np.linspace(lo, hi, max(int((hi - lo) / 0.05) + 1, 2))
        free = np.ones(len(grid), bool)
        for d, w in blocked:
            free &= np.abs(grid - d) >= (w + width) / 2 + ENTRY_MARGIN_M
        if not free.any():
            return wanted                   # no room anywhere: enter as seen, separating sorts it out
        return float(grid[free][np.argmin(np.abs(grid[free] - wanted))])

    def sample_driver(self, kind, arrival, rng, right, left, width):
        p = self.params.values
        u_speed, u_pos = rng.random(2)                  # always two draws, whatever the parameters
        base = float(self.wanted_speed(np.array(arrival.route), np.array(0.0)))
        factor = float(np.exp(p["spread"] * ndtri(u_speed))) * (1.0 if kind in TWO_WHEEL else p["large"])
        if arrival.v is not None and arrival.v > 0.5:
            factor = (arrival.v / base) ** p["keep"] * factor ** (1 - p["keep"])
        if arrival.v0 is not None:
            factor = arrival.v0 / base
        factor = float(np.clip(factor, *FACTOR_RANGE))
        lo, hi = right + width / 2, left - width / 2
        seen_at = float(arrival.d) if arrival.d is not None else lo + (hi - lo) * (0.15 + 0.7 * u_pos)
        length = DIMENSIONS[kind][0]
        d = self._free_offset(arrival.route, seen_at, length, width, right, left)
        self._entering.append((arrival.route, d, width))
        return {"factor": factor, "d": d, "pref": (seen_at - right) / max(left - right, 1e-6),
                "v": float(arrival.v) if arrival.v is not None else factor * base}

    def enter(self, slot, driver):
        self._factor[slot], self._pref[slot] = driver["factor"], driver["pref"]
        self._key[slot] = self._entered         # smaller: more senior
        self._entered += 1

    # ------------------------------------------------------------ deciding

    def decide(self, view: View) -> Decision:
        p = self.params.values
        A, I, J = view.slots, view.I, view.J
        v, d = view.v, view.d
        key = self._key[A]
        self._seen = (view.xy, view.L, view.W)
        self._entering = []

        # Which vehicles are in my line: now, or within LOOK_S at their sideways speed.
        across = np.sum(view.vel[J] * view.normal[I], axis=1)
        x0 = view.D - d[I]
        x1 = x0 + across * LOOK_S
        nearest = np.where(x0 * x1 <= 0, 0.0, np.minimum(np.abs(x0), np.abs(x1)))
        in_line = view.ok & (view.gap > 0) & (nearest - (view.half[I] + view.lat_ext) < p["clat"])

        # A junior vehicle ahead in my line and going the same way leads: exchange seniority (nearest first).
        lead = np.flatnonzero(in_line & (view.gap < SWAP_RANGE_M) & (view.heading_dot > 0.7) & (key[J] > key[I]))
        for r in lead[np.argsort(view.gap[lead], kind="stable")]:
            i, j = I[r], J[r]
            if key[j] > key[i]:
                key[i], key[j] = key[j], key[i]
        self._key[A] = key

        # Senior vehicles in the way. One coming towards me cannot be waited for: I move aside instead.
        senior = in_line & (key[J] < key[I]) & (view.gap < view.horizon[I])
        oncoming = senior & (view.heading_dot < -0.5)
        in_way = senior & ~oncoming
        # Gipps: the speed from which I can still stop behind it if it brakes.
        bt = BRAKING * p["T"]
        room = np.maximum(view.gap - p["s0"], 0.0)
        safe = -bt + np.sqrt(bt ** 2 + np.maximum(view.v_long, 0.0) ** 2 + 2 * BRAKING * room)
        allowed = np.minimum.reduceat(np.where(in_way, safe, np.inf), view.starts)

        if view.signal_state is not None:               # a red light is a standing senior vehicle
            stop = (view.signal_gap > -0.5) & ((view.signal_state == 2)
                                               | ((view.signal_state == 1) & (v ** 2 / (2 * BRAKING) < view.signal_gap)))
            at_line = -bt + np.sqrt(bt ** 2 + 2 * BRAKING * np.maximum(view.signal_gap - 0.5, 0.0))
            allowed = np.where(stop, np.minimum(allowed, at_line), allowed)

        # Separate: a junior body touching a senior one brakes if the senior one is ahead of it.
        span = view.L[I] + 2 * view.long_ext
        side_gap = np.abs(view.D - d[I]) - (view.half[I] + view.lat_ext)
        touching = view.ok & (key[J] < key[I]) & (view.gap < TOUCH_MARGIN_M) & (view.gap > -span) \
            & (side_gap < TOUCH_MARGIN_M)
        # Braking only for one in line with me; one touching my side is steered away from, not stopped for.
        ahead_touch = touching & (view.dS > 0) & (side_gap < -IN_LINE_OVERLAP_M)
        allowed = np.where(np.add.reduceat(ahead_touch.astype(np.int64), view.starts) > 0, 0.0, allowed)

        wanted = self._factor[A] * self.wanted_speed(view.route, view.s)
        acc = np.clip((np.minimum(wanted, allowed) - v) / RELAX_S, -HARD_BRAKING, ACCELERATION)

        right, left = view.net.edges(view.route, view.s)
        target = right + self._pref[A] * (left - right)
        vd = np.clip(target - d, -SIDE_SPEED, SIDE_SPEED) * np.minimum(v / 2.0, 1.0)
        # Look aside: no drift towards anyone beside me (overlapping along the road, close across it).
        beside = view.ok & (view.gap < 0.5) & (view.gap > -span) & (side_gap < BESIDE_MARGIN_M)
        to_left = np.add.reduceat((beside & (view.D > d[I])).astype(np.int64), view.starts) > 0
        to_right = np.add.reduceat((beside & (view.D < d[I])).astype(np.int64), view.starts) > 0
        vd = np.where((vd > 0) & to_left, 0.0, vd)
        vd = np.where((vd < 0) & to_right, 0.0, vd)
        # Aside from the nearest oncoming senior vehicle: away from its line, to the right when dead ahead.
        rows = np.flatnonzero(oncoming & (view.gap < DODGE_RANGE_M))
        if len(rows):
            first = {}
            for r in rows[np.argsort(-view.gap[rows], kind="stable")]:
                first[I[r]] = r                                   # the nearest one wins
            i = np.fromiter(first.keys(), np.int64)
            r = np.fromiter(first.values(), np.int64)
            away = np.where(d[i] - view.D[r] > 0.05, 1.0, -1.0)
            lo, hi = right[i] + view.half[i], left[i] - view.half[i]
            room = np.where(away > 0, hi - d[i], d[i] - lo)
            vd[i] = away * SIDE_SPEED * np.clip(room / 0.3, 0.0, 1.0)
        # Separate sideways from the touching senior vehicle nearest across; through the road edge only
        # when there is no room on the other side.
        rows = np.flatnonzero(touching)
        if len(rows):
            first = {}
            for r in rows[np.argsort(-np.abs(view.D[rows] - d[I[rows]]), kind="stable")]:
                first[I[r]] = r
            i = np.fromiter(first.keys(), np.int64)
            r = np.fromiter(first.values(), np.int64)
            lo, hi = right[i] + view.half[i], left[i] - view.half[i]
            away = np.where(d[i] - view.D[r] > 0.0, 1.0, -1.0)
            room = np.where(away > 0, hi - d[i], d[i] - lo)
            away = np.where(room < 0.2, -away, away)
            vd[i] = away * SEPARATE_SPEED
        return Decision(acc=acc, vd=vd)
