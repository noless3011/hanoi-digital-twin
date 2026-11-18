"""Flow model: a two-dimensional Intelligent Driver Model with gap acceptance at crossings. 8 parameters.

The choice model scores ~30 candidate actions against seven hand-weighted costs. This one uses three
textbook rules, each with parameters that have a physical meaning:

  following  IDM (Treiber, Hennecke and Helbing 2000), with every vehicle ahead as a leader, weighted by
             how much it is in line with the driver:
                 acc = a [1 - (v/v0)^4 - max_j w_j (s*_j / gap_j)^2]
                 s*_j = s0 + v T + v (v - v_j) / (2 sqrt(a b))
             w_j is 1 when the bodies overlap sideways and falls linearly to 0 at a sideways clearance
             of clat, so a motorcycle can sit staggered beside the one ahead and pass through a gap
             wider than its body plus clat.
  sideways   among a few positions across the road around where the driver is (OFFSETS, as far as it
             can drift in TAU_LAT), it heads for the one where IDM would let it accelerate most, less
             w_pos per m2 away from its own place across the road, a cost for leaving the road and for
             running into someone alongside. Filtering past slower vehicles and moving aside for
             oncoming ones both come out of this (MOBIL's idea, on a continuous road).
  crossings  a vehicle on a crossing path (headings more than 45 degrees apart, but not within 25 of head-on) conflicts with the
             driver if it will be in the driver's way at any time from t_gap before the driver reaches
             it to t_gap after the driver has passed. Of two conflicting vehicles, the one that would
             get there first goes and the other treats the crossing point as a standing vehicle.
             Waiting earns priority (WAIT_PRIORITY seconds per second), so nobody waits for ever.

Signals: a red stop line, or an amber one the driver can still stop for, is a standing vehicle.

Desired speed is as in the choice model: Arrival.v0 if given, else v0_seen times the speed seen on
video, else drawn around V0_FREE. Each driver's time gap is T times its own caution (log-normal,
CAUTION_SPREAD). Three random draws per driver, whatever the parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .base import BehaviourModel, Decision, View
from .params import RATIOS, ParamSet, Spec
from .registry import register

SPECS = (
    Spec("v0_seen", 1.1, 0.7, 1.6, True, 1, "desired speed as a multiple of the speed seen on video"),
    Spec("a", 1.8, 0.5, 4.0, True, 1, "IDM: largest acceleration (m/s2)"),
    Spec("b", 2.5, 1.0, 5.0, True, 1, "IDM: comfortable braking (m/s2)"),
    Spec("T", 0.8, 0.1, 2.5, True, 1, "IDM: time gap kept to a vehicle in line ahead (s)"),
    Spec("s0", 0.8, 0.2, 3.0, True, 1, "IDM: gap kept at standstill (m)"),
    Spec("clat", 0.4, 0.05, 1.5, True, 1, "sideways clearance beyond which a vehicle ahead is not in line (m)"),
    Spec("w_pos", 0.2, 0.01, 3.0, True, 1, "pull back to the driver's own place across the road (m/s2 per m2)"),
    Spec("t_gap", 1.5, 0.2, 5.0, True, 1, "time margin around a crossing vehicle's passage (s)"),
)
NAMES = tuple(s.name for s in SPECS)
FLOW_RATIOS = {k: {n: r for n, r in v.items() if n in NAMES} for k, v in RATIOS.items()}
V0_RATIO = {k: v.get("v0", 1.0) for k, v in RATIOS.items()}

OFFSETS = np.array([-1.2, -0.5, 0.0, 0.5, 1.2])   # sideways positions considered, from where the driver is (m)
CENTRE = 2
TAU_LAT = 1.0             # time to reach the chosen sideways position; others' sideways motion projected as far
VLAT = 0.8                # fastest sideways drift (m/s)
MAX_SIDE_SLIP = 0.5       # sideways speed is at most this share of forward speed,
SHUFFLE = 0.2             #   plus this much when standing (m/s)
V0_FREE, V0_SPREAD = 8.0, 0.15        # desired speed of a vehicle not seen on video (m/s)
CAUTION_SPREAD = 0.3
DELTA = 4                 # IDM free-road exponent
MAX_BRAKE = 9.0           # m/s2
MAX_TERM = 50.0           # cap of an IDM interaction term (keeps the numbers finite at tiny gaps)
EDGE = 5.0                # cost of leaving the road (m/s2 per m2)
ALONGSIDE = 4.0           # cost of drifting into a vehicle alongside (m/s2 at full overlap)
STAY = 0.05               # bonus for holding the current position (stops flicker, m/s2)
LATERAL_REACH_M = 14.0
CROSS_DOT = 0.7           # paths cross when the headings are more than 45 degrees apart,
ONCOMING_DOT = 0.9        #   unless within 25 degrees of head-on: then the vehicles meet and pass
MIN_SPEED = 1.0           # floor of the speeds in times to reach a crossing (m/s)
WAIT_PRIORITY = 0.3       # seconds of priority per second waited
WAIT_SPEED = 0.5
ANTICIPATE_S = 4.0        # time before meeting an oncoming vehicle at which a driver starts moving aside (s)
OVERLAP_M = 0.3           # sideways overlap at which an oncoming vehicle is fully in the way (m)
BREAK_S = 2.0             # standing time after which a deadlock is broken (s)
MAX_W = 3.0               # largest in-line weight (deep overlap)


@register("flow")
class FlowModel(BehaviourModel):
    SPECS = SPECS
    PER_SLOT = NAMES + ("v0", "pref", "wait")

    @classmethod
    def default_params(cls, scene=None):
        return ParamSet(ratios=FLOW_RATIOS, specs=SPECS, model=cls.name)

    def __init__(self, params=None):
        if not isinstance(params, ParamSet):
            params = ParamSet(params, FLOW_RATIOS, SPECS, self.name)
        self.params = params
        self._by_kind = {}

    def resize(self, capacity, n_old):
        for name in self.PER_SLOT:
            new = np.zeros(capacity)
            if n_old:
                new[:n_old] = getattr(self, "_" + name)[:n_old]
            setattr(self, "_" + name, new)

    def sample_driver(self, kind, arrival, rng, right, left, width):
        p = self._by_kind.get(kind)
        if p is None:
            p = self._by_kind[kind] = self.params.of(kind)
        u_speed, u_caution, u_pos = rng.random(3)
        driver = dict(p)
        if arrival.v0 is not None:
            driver["v0"] = float(arrival.v0)
        elif arrival.v_seen is not None:
            driver["v0"] = p["v0_seen"] * float(arrival.v_seen)
        else:
            driver["v0"] = V0_FREE * V0_RATIO.get(kind, 1.0) * float(np.exp(V0_SPREAD * ndtri(u_speed)))
        driver["T"] = p["T"] * float(np.exp(CAUTION_SPREAD * ndtri(u_caution)))
        lo, hi = right + width / 2, left - width / 2
        d = float(arrival.d) if arrival.d is not None else lo + (hi - lo) * (0.15 + 0.7 * u_pos)
        driver["d"] = d
        driver["pref"] = (d - right) / max(left - right, 1e-6)
        driver["v"] = float(arrival.v) if arrival.v is not None else driver["v0"]
        return driver

    def enter(self, slot, driver):
        for name in NAMES + ("v0", "pref"):
            getattr(self, "_" + name)[slot] = driver[name]
        self._wait[slot] = 0.0

    def decide(self, view: View) -> Decision:
        A = view.slots
        n = len(A)
        if self.debug:
            self._view = view
        P = {name: getattr(self, "_" + name)[A] for name in self.PER_SLOT}
        v, d = view.v, view.d
        a, b, T, s0, clat = P["a"], P["b"], P["T"], P["s0"], P["clat"]
        sqab = 2 * np.sqrt(a * b)
        K = len(OFFSETS)

        # ---- sideways positions within reach
        reach = np.minimum(VLAT, SHUFFLE + MAX_SIDE_SLIP * v) * TAU_LAT
        dk = d[:, None] + np.clip(OFFSETS[None, :], -reach[:, None], reach[:, None])         # (n, K)
        right, left = view.net.edges(view.route, view.s + v * TAU_LAT)
        lo, hi = right + view.half, left - view.half
        target = right + P["pref"] * (left - right)

        inter = np.zeros((n, K))       # largest weighted IDM interaction term at each position
        inter_long = np.zeros(n)       # the term the driver brakes for, where it is
        oncoming = np.zeros((n, K))    # how much each position is in the way of someone coming the other way
        along = np.zeros((n, K))       # overlap with vehicles alongside at each position
        rows = np.flatnonzero(view.ok & (view.gap < view.horizon[view.I])
                              & (np.abs(view.D - d[view.I]) < LATERAL_REACH_M))
        if len(rows):
            i, j = view.I[rows], view.J[rows]
            ext = view.half[i] + view.lat_ext[rows]
            vj_lat = np.sum(view.vel[j] * view.normal[i], axis=1)
            D_next = view.D[rows] + vj_lat * TAU_LAT
            side_now = np.abs(view.D[rows] - d[i]) - ext
            side_k = np.abs(D_next[:, None] - dk[i]) - ext[:, None]                         # (R, K)
            # beyond 1 when the bodies overlap, so overlapping drivers have a side to move apart to
            w_k = np.clip(1 - side_k / clat[i][:, None], 0, MAX_W)
            w_now = np.clip(1 - side_now / clat[i], 0, MAX_W)
            w_k[:, CENTRE] = np.maximum(w_k[:, CENTRE], w_now)
            span = view.L[i] / 2 + view.long_ext[rows]
            ahead = view.dS[rows] > 0
            gap = np.maximum(view.gap[rows], 0.1)
            vi = v[i]

            # following: every vehicle ahead, weighted by how much it is in line
            # An oncoming vehicle is passed, not followed: no closing-speed term, and it is in the way
            # only where the bodies would overlap (OVERLAP_M of overlap counts fully).
            opposed = view.heading_dot[rows] < -ONCOMING_DOT
            dv = np.where(opposed, 0.0, vi - view.v_long[rows])
            s_star = s0[i] + np.maximum(vi * T[i] + vi * dv / sqab[i], 0)
            term = np.minimum((s_star / gap) ** 2, MAX_TERM)
            w_on = np.clip(-side_k / OVERLAP_M, 0, MAX_W)
            w_on[:, CENTRE] = np.maximum(w_on[:, CENTRE], np.clip(-side_now / OVERLAP_M, 0, MAX_W))
            w_ramp = w_k
            w_k = np.where(opposed[:, None], w_on, w_k)
            fr = np.flatnonzero(ahead)
            np.maximum.at(inter, i[fr], w_k[fr] * term[fr, None])
            # Oncoming in line: move aside well before meeting (IDM alone would only brake, late).
            # Each driver moves away from the other, so both keep to their own side of the meeting.
            on = np.flatnonzero(ahead & opposed)
            if len(on):
                t_meet = view.gap[rows][on] / np.maximum(vi[on] - view.v_long[rows][on], MIN_SPEED)
                soon = np.clip(1 - t_meet / ANTICIPATE_S, 0, 1)
                np.add.at(oncoming, i[on], np.minimum(w_ramp[on], 1.0) * soon[:, None])
            # Deadlock breaker: two vehicles not going the same way, both standing, each in the other's
            # way. After BREAK_S the one that has waited longer stops braking for the other (it still
            # steers round it, through the sideways choice); the other keeps waiting.
            wi, wj = P["wait"][i], P["wait"][j]
            senior = (wi > wj) | ((wi == wj) & (view.uid[i] < view.uid[j]))
            breaks = (ahead & (view.heading_dot[rows] <= CROSS_DOT) & (vi < WAIT_SPEED) & (v[j] < WAIT_SPEED)
                      & (wi > BREAK_S) & senior)
            lr = np.flatnonzero(ahead & ~breaks)
            np.maximum.at(inter_long, i[lr], np.minimum(w_k[lr, CENTRE], 1.0) * term[lr])
            if self.debug:
                self.dbg = {"follow": inter[:, CENTRE].copy(), "rows": rows, "term": term, "w": w_k[:, CENTRE],
                            "ahead": ahead, "cross": np.zeros(n), "yield_to": {}}

            # alongside: bodies level with each other (j's centre not ahead); do not drift into them
            ar = np.flatnonzero(~ahead & (view.dS[rows] > -span))
            np.maximum.at(along, i[ar], w_k[ar])

            # crossings
            hd = view.heading_dot[rows]
            cr = np.flatnonzero(ahead & (hd <= CROSS_DOT) & ~opposed & (view.gap[rows] > 0))
            if len(cr):
                ic, jc = i[cr], j[cr]
                hn = np.sum(view.heading[jc] * view.normal[ic], axis=1)          # j's heading across i's route
                towards = (view.D[rows][cr] - d[ic]) * hn < 0
                sn = side_now[cr]
                lat_speed = np.maximum(np.abs(vj_lat[cr]), MIN_SPEED)
                t_j_in = np.maximum(sn, 0) / lat_speed
                t_j_out = (sn + 2 * ext[cr]) / lat_speed
                vi_c = np.maximum(vi[cr], MIN_SPEED)
                t_i_in = view.gap[rows][cr] / vi_c
                t_i_out = (view.gap[rows][cr] + 2 * span[cr]) / vi_c
                tg = P["t_gap"][ic]
                conflict = (towards | (sn <= 0)) & (t_j_in < t_i_out + tg) & (t_j_out > t_i_in - tg)
                first = (t_j_in - WAIT_PRIORITY * P["wait"][jc]) < (t_i_in - WAIT_PRIORITY * P["wait"][ic])
                # Both standing: the times to get there are guesses each side makes differently, so
                # both could wait. Then the one that has waited longer goes; both see it the same way.
                both = (vi[cr] < WAIT_SPEED) & (v[jc] < WAIT_SPEED)
                first = np.where(both, ~senior[cr], first)
                y = np.flatnonzero(conflict & first)
                if len(y):
                    iy = ic[y]
                    gy = gap[cr][y]
                    vy = v[iy]
                    s_stop = s0[iy] + vy * T[iy] + vy * vy / sqab[iy]
                    stop = np.minimum((s_stop / gy) ** 2, MAX_TERM)
                    np.maximum.at(inter, iy, np.repeat(stop[:, None], K, axis=1))
                    np.maximum.at(inter_long, iy, stop)
                    if self.debug:
                        np.maximum.at(self.dbg["cross"], iy, stop)
                        for a_, b_ in zip(iy, jc[y]):
                            self.dbg["yield_to"].setdefault(int(a_), []).append(int(b_))

        # ---- red and amber stop lines
        if view.signal_state is not None:
            g0 = view.signal_gap
            must = (g0 > -0.5) & ((view.signal_state == 2) | ((view.signal_state == 1) & (v * v / (2 * b) < g0)))
            if must.any():
                g = np.maximum(g0, 0.1)
                stop = np.where(must, np.minimum(((s0 * 0.5 + v * T + v * v / sqab) / g) ** 2, MAX_TERM), 0.0)
                inter = np.maximum(inter, stop[:, None])
                inter_long = np.maximum(inter_long, stop)

        # ---- choose
        acc_k = a[:, None] * (1 - (v / P["v0"])[:, None] ** DELTA - inter)
        outside = np.maximum(lo[:, None] - dk, 0) + np.maximum(dk - hi[:, None], 0)
        outside_now = np.maximum(lo - d, 0) + np.maximum(d - hi, 0)
        util = (acc_k - P["w_pos"][:, None] * (dk - target[:, None]) ** 2
                - EDGE * np.maximum(outside - outside_now[:, None], 0) ** 2 - EDGE * 0.2 * outside ** 2
                - ALONGSIDE * along - a[:, None] * oncoming)
        util[:, CENTRE] += STAY
        k = np.argmax(util, axis=1)
        row = np.arange(n)
        slip = np.minimum(VLAT, SHUFFLE + MAX_SIDE_SLIP * v)
        vd = np.clip((dk[row, k] - d) / TAU_LAT, -slip, slip)
        acc = np.clip(a * (1 - (v / P["v0"]) ** DELTA - inter_long), -MAX_BRAKE, a)
        acc = np.maximum(acc, -v / view.dt)       # do not reverse
        return Decision(acc=acc, vd=vd)

    def after_move(self, view: View, v_new):
        A = view.slots
        self._wait[A] = np.where(v_new < WAIT_SPEED, self._wait[A] + view.dt, np.maximum(self._wait[A] - 2 * view.dt, 0))
