"""Choice model: each step every driver compares a few actions and takes the cheapest.

An action is an acceleration and a sideways speed. For each one the driver predicts where it and the
vehicles around it will be over the next `tau` seconds (others keep their velocity) and adds up costs:

  progress   ((speed reached - desired speed) / desired speed)^2, weighted more the longer it has waited
  risk       closeness to each other vehicle at the predicted positions: 0 when the gap ahead is above
             s0 + T * speed or the sideways gap above clat, rising to 1 as the gap closes and the bodies
             line up, more when they would overlap. Vehicles wholly behind are their own drivers' business.
  open road  being behind something slower: the share of desired speed the way ahead does not allow at
             the sideways position the action leads to (same weight as progress)
  keep clear coming to rest in the way of a vehicle that is waiting to cross (the road just ahead of
             it); whoever has waited longer counts for more
  position   distance from the driver's own position across the road, and leaving the road
  effort     hard acceleration, braking and sideways movement
  stop line  running a red light (when the simulation has signals)

Following and giving way at crossings come from the risk term; filtering past slower vehicles and
easing aside for oncoming traffic from risk and open road together. There are no separate rules. Parameters are in params.py. With no other
vehicle near, a driver holds its desired speed and its position: the constant-speed model.

Desired speed: given (Arrival.v0); else, for a vehicle seen on video, v0_seen times the speed it was
seen at when not held up (Arrival.v_seen, measure/demand.py), so each replayed driver keeps its own pace;
else drawn around v0.

Randomness: a driver's desired speed and caution are drawn once, at entry, from the engine's
generator. The draws do not depend on the parameters, so two runs with the same seed and demand differ
only by their parameters.

    result = Simulation(network, ChoiceModel(ParamSet()), arrivals, seed=1).run(180)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .base import BehaviourModel, Decision, View
from .params import NAMES, RATIOS, SPECS, ParamSet
from .registry import register

ACC_LEVELS = 6
LATERAL = np.array([-1.0, -0.3, 0.0, 0.3, 1.0])     # sideways speed as a share of vlat
TIME_SAMPLES = 5                # prediction points over tau
HIT_COST = 3.0                  # added risk when bodies would overlap
EFFORT = 0.05                   # weight of effort
STICK = 0.03                    # bonus for repeating the last action (stops flicker)
EDGE = 5.0                      # cost per m2 beyond the road edge
CAUTION_SPREAD = 0.3            # spread of caution between drivers (log-normal sigma on w_risk)
MAX_SIDE_SLIP = 0.5             # sideways speed is at most this share of forward speed,
SHUFFLE = 0.2                   #   plus this much when standing (m/s)
RELAX_S = 0.5                   # time to close the last gap to the desired speed
WAIT_SPEED = 0.5                # below this a driver is waiting (m/s)
MAX_IMPATIENCE = 5.0
LATERAL_REACH_M = 14.0          # vehicles further aside than this are not considered
STANDING_SPEED = 1.0            # slower than this, a vehicle is an obstacle to steer round (m/s)
KEEP_CLEAR = 2.0                # weight of coming to rest in a waiting crosser's way (times w_risk)
KEEP_CLEAR_M = 4.0              # how far ahead of a waiting crosser is kept clear (m)
CROSSER_SPEED = 2.0             # a crossing vehicle slower than this is waiting (m/s)
REST_SPEED = 1.0                # ending an action slower than this is coming to rest (m/s)


@register("choice")
class ChoiceModel(BehaviourModel):
    SPECS = SPECS

    @classmethod
    def default_params(cls, scene=None):
        return ParamSet()

    def __init__(self, params=None):
        self.params = params if isinstance(params, ParamSet) else ParamSet(params)
        self._by_kind = {}

    # ------------------------------------------------------------ drivers

    PER_SLOT = NAMES + ("pref", "caution", "wait", "last")

    def resize(self, capacity, n_old):
        for name in self.PER_SLOT:
            new = np.zeros(capacity) if name != "last" else np.full(capacity, -1, np.int64)
            if n_old:
                new[:n_old] = getattr(self, "_" + name)[:n_old]
            setattr(self, "_" + name, new)

    def sample_driver(self, kind, arrival, rng, right, left, width):
        p = self._by_kind.get(kind)
        if p is None:
            p = self._by_kind[kind] = self.params.of(kind)
        u_speed, u_caution, u_pos = rng.random(3)        # always three draws, whatever the parameters
        driver = dict(p)
        if arrival.v0 is not None:
            driver["v0"] = float(arrival.v0)
        elif arrival.v_seen is not None:     # a video vehicle: its own free speed, scaled
            driver["v0"] = p["v0_seen"] * float(arrival.v_seen)
        else:
            driver["v0"] = p["v0"] * float(np.exp(p["v0_spread"] * ndtri(u_speed)))
        driver["caution"] = float(np.exp(CAUTION_SPREAD * ndtri(u_caution)))
        lo, hi = right + width / 2, left - width / 2
        d = float(arrival.d) if arrival.d is not None else lo + (hi - lo) * (0.15 + 0.7 * u_pos)
        driver["d"] = d
        # Position across the road as a share of its width, so it follows the road as it widens.
        driver["pref"] = (d - right) / max(left - right, 1e-6)
        driver["v"] = float(arrival.v) if arrival.v is not None else driver["v0"]
        return driver

    def enter(self, slot, driver):
        for name in NAMES + ("pref", "caution"):
            getattr(self, "_" + name)[slot] = driver[name]
        self._wait[slot] = 0.0
        self._last[slot] = -1

    # ------------------------------------------------------------ deciding

    def decide(self, view: View) -> Decision:
        A = view.slots
        n = len(A)
        P = {name: getattr(self, "_" + name)[A] for name in self.PER_SLOT}
        v, d, tau = view.v, view.d, P["tau"]
        v0, a, b = P["v0"], P["a"], P["b"]

        # ---- candidate actions: (acceleration, speed it stops changing at) x sideways speed
        zero = np.zeros(n)
        acc6 = np.stack([-2 * b, -b, -0.4 * b, zero, 0.5 * a, a], axis=1)                      # (n, 6)
        lim6 = np.stack([zero, zero, np.where(v > v0, v0, 0.0), v, np.maximum(v, v0), np.maximum(v, v0)], axis=1)
        acc = np.repeat(acc6, len(LATERAL), axis=1)                                             # (n, K)
        lim = np.repeat(lim6, len(LATERAL), axis=1)
        K = acc.shape[1]
        t = tau[:, None] * (np.arange(1, TIME_SAMPLES + 1) / TIME_SAMPLES)[None, :]             # (n, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_change = np.where(acc != 0, (lim - v[:, None]) / acc, np.inf)                     # (n, K) >= 0
        tt = np.minimum(t[:, None, :], t_change[:, :, None])                                    # (n, K, M)
        ds = v[:, None, None] * tt + acc[:, :, None] * tt ** 2 / 2 + lim[:, :, None] * (t[:, None, :] - tt)
        v_t = v[:, None, None] + acc[:, :, None] * tt                                           # (n, K, M)
        v_end = v_t[:, :, -1]
        v_mean = ds[:, :, -1] / tau[:, None]
        vd = np.tile(LATERAL, ACC_LEVELS)[None, :] * P["vlat"][:, None]
        slip = SHUFFLE + MAX_SIDE_SLIP * v_mean
        vd = np.clip(vd, -slip, slip)                                                           # (n, K)
        d_t = d[:, None, None] + vd[:, :, None] * t[:, None, :]                                 # (n, K, M)

        # ---- progress, position, effort
        urge = np.minimum(1 + P["wait"] / P["T_imp"], MAX_IMPATIENCE)
        cost = urge[:, None] * ((v_end - v0[:, None]) / v0[:, None]) ** 2
        right, left = view.net.edges(view.route, view.s + v * tau / 2)
        lo, hi = right + view.half, left - view.half
        target = right + P["pref"] * (left - right)
        d_end = d_t[:, :, -1]
        cost += P["w_pos"][:, None] * (d_end - target[:, None]) ** 2
        outside = np.maximum(lo[:, None] - d_end, 0) + np.maximum(d_end - hi[:, None], 0)
        # Already outside (entered there, or the road narrowed): only moving further out costs.
        outside_now = np.maximum(lo - d, 0) + np.maximum(d - hi, 0)
        cost += EDGE * np.maximum(outside - outside_now[:, None], 0) ** 2 + EDGE * 0.2 * outside ** 2
        cost += EFFORT * (((v_end - v[:, None]) / tau[:, None] / b[:, None]) ** 2 + (vd / P["vlat"][:, None]) ** 2)

        # ---- risk from each nearby vehicle, at the predicted positions
        span = view.L[view.I] + 2 * view.long_ext
        rows = np.flatnonzero(view.ok & (view.gap > -span) & (view.gap < view.horizon[view.I])
                              & (np.abs(view.D - d[view.I]) < LATERAL_REACH_M))
        if len(rows):
            i, j = view.I[rows], view.J[rows]
            vj_lat = np.sum(view.vel[j] * view.normal[i], axis=1)
            ti = t[i][:, None, :]                                                                # (P, 1, M)
            gap = view.gap[rows][:, None, None] + view.v_long[rows][:, None, None] * ti - ds[i]  # (P, K, M)
            side = np.abs(view.D[rows][:, None, None] + vj_lat[:, None, None] * ti - d_t[i]) \
                - (view.half[i] + view.lat_ext[rows])[:, None, None]
            wanted = P["s0"][i][:, None, None] + P["T"][i][:, None, None] * v_t[i]
            along = gap > -span[rows][:, None, None]            # not wholly behind
            p_long = np.clip(1 - gap / wanted, 0, 1) * along
            # Sideways closeness: 1 with the narrower body wholly in line, falling steadily to 0 at the
            # wanted clearance, so every bit of sideways room is worth something (staggered following).
            clat = P["clat"][i][:, None, None]
            narrow = (2 * np.minimum(view.half[i], view.lat_ext[rows]))[:, None, None]
            p_side = np.clip((clat - side) / (clat + narrow), 0, 1)
            # Bodies overlapping: worse the deeper, so drivers already in contact move apart.
            depth = np.minimum(np.minimum(-gap, gap + span[rows][:, None, None]), -side)
            risk = p_long ** 2 * p_side + HIT_COST * np.where(depth > 0, 1 + depth, 0.0)
            total = np.zeros((n, K))
            np.add.at(total, i, risk.max(axis=2))
            cost += (P["w_risk"] * P["caution"])[:, None] * total

            # ---- open road: how fast the way ahead allows at each sideways position. A slower (or
            # oncoming) vehicle in line ahead caps it; the cap eases off as the bodies come out of line.
            # This is what makes a driver pull out to pass, or ease aside, before the gap gets short.
            # Vehicles going the same or the opposite way count, and anything standing, whichever way it
            # faces (two vehicles stopped across each other's paths have to steer round, not wait).
            standing = np.linalg.norm(view.vel[j], axis=1) < STANDING_SPEED
            lined = np.flatnonzero((view.gap[rows] > 0) & ((np.abs(view.heading_dot[rows]) > 0.7) | standing))
            if len(lined):
                r, il = rows[lined], i[lined]
                cap = np.maximum(view.v_long[r], 0) + view.gap[r] / (tau[il] + P["T"][il])        # (Q,)
                short = np.clip((v0[il] - cap) / v0[il], 0, 1)                                     # 0: no hindrance
                side_end = np.abs(view.D[r][:, None] - d_end[il]) - (view.half[il] + view.lat_ext[r])[:, None]
                clat_q = P["clat"][il][:, None]
                narrow_q = (2 * np.minimum(view.half[il], view.lat_ext[r]))[:, None]
                blocked = short[:, None] * np.clip((clat_q - side_end) / (clat_q + narrow_q), 0, 1)   # (Q, K)
                worst = np.zeros((n, K))
                np.maximum.at(worst, il, blocked)
                cost += urge[:, None] * worst ** 2

            # ---- keep clear: do not come to rest in the way of a vehicle waiting to cross. Its way is the
            # KEEP_CLEAR_M of road in front of it. Driving through is fine; so is being there already.
            # Whoever has waited longer counts for more, so of two drivers at each other's paths one goes.
            hd = view.heading_dot[rows]
            cross = np.flatnonzero((np.abs(hd) <= 0.7) & (np.linalg.norm(view.vel[j], axis=1) < CROSSER_SPEED))
            if len(cross):
                r, ic, jc, hd = rows[cross], i[cross], j[cross], hd[cross]
                hn = np.sum(view.heading[jc] * view.normal[ic], axis=1)          # j's heading across i's route
                reach = view.L[jc] / 2 + KEEP_CLEAR_M / 2
                z_s, z_d = view.dS[r] + reach * hd, view.D[r] + reach * hn      # middle of j's way, seen from i
                z_long = np.abs(hd) * KEEP_CLEAR_M / 2 + np.abs(hn) * view.W[jc] / 2 + view.L[ic] / 2
                z_side = np.abs(hn) * KEEP_CLEAR_M / 2 + np.abs(hd) * view.W[jc] / 2 + view.half[ic]
                there_now = (np.abs(z_s) < z_long) & (np.abs(z_d - d[ic]) < z_side)
                there = (np.abs(z_s[:, None] - ds[ic][:, :, -1]) < z_long[:, None])                     & (np.abs(z_d[:, None] - d_end[ic]) < z_side[:, None])
                resting = there & (v_end[ic] < REST_SPEED) & ~there_now[:, None]
                weight = KEEP_CLEAR * urge[jc] / urge[ic]
                worst = np.zeros((n, K))
                np.maximum.at(worst, ic, resting * weight[:, None])
                cost += (P["w_risk"] * P["caution"])[:, None] * worst

        # ---- red and amber stop lines: a standing obstacle across the road
        if view.signal_state is not None:
            g0 = view.signal_gap
            stop_distance = v ** 2 / (2 * b)
            must = (g0 > -0.5) & ((view.signal_state == 2) | ((view.signal_state == 1) & (stop_distance < g0)))
            if must.any():
                g = g0[:, None, None] - ds
                over = np.clip(1 - g / (0.5 + 0.3 * v_t), 0, 1) ** 2 + HIT_COST * (g < 0)
                cost += must[:, None] * P["w_risk"][:, None] * over.max(axis=2)

        # ---- choose
        last = P["last"].astype(np.int64)
        has_last = last >= 0
        cost[np.flatnonzero(has_last), last[has_last]] -= STICK
        k = np.argmin(cost, axis=1)
        self._last[A] = k
        row = np.arange(n)
        # The chosen acceleration, stopping at its speed limit (easing in over RELAX_S near the desired speed).
        room = lim[row, k] - v
        chosen = acc[row, k]
        acc_out = np.where(chosen > 0, np.minimum(chosen, np.maximum(room, 0) / RELAX_S),
                           np.where(chosen < 0, np.maximum(chosen, np.minimum(room, 0) / view.dt), 0.0))
        return Decision(acc=acc_out, vd=vd[row, k])

    def after_move(self, view: View, v_new):
        A = view.slots
        self._wait[A] = np.where(v_new < WAIT_SPEED, self._wait[A] + view.dt, np.maximum(self._wait[A] - 2 * view.dt, 0))


# Parameters the Morris screening (runs/twin/sensitivity-choice-v2: observed speeds, 12 clips, 16
# trajectories) found to move the score or the contacts by at least twice their noise floor. The rest
# (v0_spread, vlat, T_imp) stay at their defaults.
CORE_NAMES = ("v0", "v0_seen", "a", "b", "T", "s0", "clat", "w_pos", "tau", "w_risk")


@register("choice-core")
class CoreChoiceModel(ChoiceModel):
    """The choice model with only its screened-in parameters free; the others are held at their defaults."""
    SPECS = tuple(s for s in SPECS if s.name in CORE_NAMES)

    @classmethod
    def default_params(cls, scene=None):
        return ParamSet(ratios=RATIOS, specs=cls.SPECS, model=cls.name)

    def __init__(self, params=None):
        if not isinstance(params, ParamSet):
            params = ParamSet(params, RATIOS, self.SPECS, self.name)
        self.core = params
        super().__init__(ParamSet({**params.values}, params.ratios))
