"""The simulation engine: vehicles on routes, deterministic for a seed, no rendering. It knows nothing
about how drivers decide; a behaviour model does that (twin/behaviour/, contract in behaviour/base.py).

Every vehicle follows its route in (s, d) coordinates: s along the route, d sideways. Each step:

  1. spawn   due arrivals enter at the route start, at the speed and sideways offset the model gives
             (the engine does not check whether there is room);
  2. see     the engine builds a View: every vehicle's pose, and every nearby vehicle projected into
             its route coordinates (gap, offset, extents, speed along the route), plus signal states;
  3. decide  the model returns accelerations and sideways speeds;
  4. move    the engine integrates exactly that; vehicles may pass through each other;
  5. record  positions are recorded, contacts between bodies scored, and finished vehicles removed.

    sim = Simulation(network, ConstantSpeedModel(), arrivals, seed=1)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .behaviour.base import BehaviourModel, View
from .vehicles import DIMENSIONS, KINDS

NEAR_FRAME_M = 8.0          # gaps to vehicles this close are also measured straight ahead
CONTACT_SHRINK = 0.1        # bodies are shrunk by this share before checking for contact


@dataclass(order=True)
class Arrival:
    t: float
    kind: str = field(compare=False)
    route: int = field(compare=False)
    uid: int = field(compare=False, default=-1)
    d: float | None = field(compare=False, default=None)   # sideways offset at entry (None: preferred)
    v: float | None = field(compare=False, default=None)   # entry speed (None: free)
    v0: float | None = field(compare=False, default=None)  # this driver's desired speed (None: sampled)
    v_seen: float | None = field(compare=False, default=None)  # speed seen on video (None: not seen)


class Simulation:
    def __init__(self, network, model, arrivals, *, seed=1, dt=0.1, signals=None, record_every=1,
                 check_collisions=True, max_vehicles=2000):
        """model: a BehaviourModel, one instance per simulation (it holds the drivers' state).
        check_collisions=False skips contact recording (faster; the contact score is then empty)."""
        if not isinstance(model, BehaviourModel):
            raise TypeError("model must be a BehaviourModel")
        self.net, self.model, self.dt = network, model, float(dt)
        self.rng = np.random.default_rng(seed)
        self.record_every, self.check_collisions = record_every, check_collisions
        self.pending = sorted(arrivals)
        for k, arrival in enumerate(self.pending):
            if arrival.uid < 0:
                arrival.uid = k
        self.next_pending = 0
        self.t, self.steps = 0.0, 0
        self.signals = signals
        if signals is not None:
            self.sig_group, self.sig_stop = signals.bind(network)
        cap = 64
        self._alloc(cap)
        self.model.bind(network, cap)
        self.vehicles = {}                  # uid -> record
        self.samples = []
        self.signal_log = []
        self.collisions = {}                # (uid, uid) -> time of first contact
        self.contacts = {}                  # (uid, uid) -> contact record (see _collisions)
        self.max_vehicles = max_vehicles
        self.max_stuck = 0.0
        self._debug = False
        self.collision_log = []

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = self.model.debug = bool(value)

    # ------------------------------------------------------------ state storage

    FIELDS = ("route", "s", "d", "v", "vd", "acc", "L", "W", "stuck", "uid", "kind")

    def _alloc(self, cap):
        old = getattr(self, "active", None)
        n_old = 0 if old is None else len(old)
        new = {name: np.zeros(cap) for name in self.FIELDS}
        new["route"] = np.zeros(cap, np.int64)
        new["uid"] = np.full(cap, -1, np.int64)
        new["kind"] = np.zeros(cap, np.int64)
        active = np.zeros(cap, bool)
        if old is not None:
            for name in self.FIELDS:
                new[name][:n_old] = getattr(self, name)
            active[:n_old] = old
        for name, value in new.items():
            setattr(self, name, value)
        self.active = active
        if old is not None:
            self.model.resize(cap, n_old)

    def _slot(self):
        free = np.flatnonzero(~self.active)
        if len(free) == 0:
            if len(self.active) >= self.max_vehicles:
                raise RuntimeError("Too many vehicles at once")
            k = len(self.active)
            self._alloc(min(2 * len(self.active), self.max_vehicles))
            return k
        return int(free[0])

    # ------------------------------------------------------------ geometry of the current state

    def _pose(self, A):
        route, s, d, v, vd = self.route[A], self.s[A], self.d[A], self.v[A], self.vd[A]
        xy, t, n = self.net.world(route, s, d)
        vel = t * v[:, None] + n * vd[:, None]
        # The body points along the route, tilted by sideways drift (a shuffling stopped bike does not turn).
        tilt = np.clip(np.arctan2(vd, np.maximum(v, 2.0)), -0.2, 0.2) * np.minimum(v / 2.0, 1.0)
        heading = t * np.cos(tilt)[:, None] + n * np.sin(tilt)[:, None]
        return xy, heading, vel, t, n

    # ------------------------------------------------------------ spawning

    def _spawn(self):
        """Every arrival that is due enters now, wherever the model puts it."""
        while self.next_pending < len(self.pending) and self.pending[self.next_pending].t <= self.t + 1e-9:
            self._place(self.pending[self.next_pending])
            self.next_pending += 1

    def _place(self, arrival):
        L, W = DIMENSIONS[arrival.kind]
        r = arrival.route
        s = L / 2
        right, left = (float(x) for x in self.net.edges(np.array(r), np.array(s)))
        driver = self.model.sample_driver(arrival.kind, arrival, self.rng, right, left, W)
        k = self._slot()
        values = {"route": r, "s": s, "d": float(driver["d"]), "v": max(float(driver["v"]), 0.0), "vd": 0.0,
                  "acc": 0.0, "L": L, "W": W, "stuck": 0.0, "uid": arrival.uid, "kind": KINDS.index(arrival.kind)}
        for name, value in values.items():
            getattr(self, name)[k] = value
        self.active[k] = True
        self.model.enter(k, driver)
        self.vehicles[arrival.uid] = {"uid": arrival.uid, "kind": arrival.kind, "route": r, "t_sched": arrival.t,
                                      "t_spawn": self.t, "t_exit": None, "v_entry": values["v"], "slot": k}

    # ------------------------------------------------------------ perception

    def view(self, A):
        """What the drivers in slots A perceive now (see behaviour.base.View)."""
        net = self.net
        n = len(A)
        r, s, d, v = self.route[A], self.s[A], self.d[A], self.v[A]
        L, W = self.L[A], self.W[A]
        xy, heading, vel, tan, nrm = self._pose(A)
        # Sideways half extent of each body as tilted by its drift.
        tilt_sin = np.abs(tan[:, 0] * heading[:, 1] - tan[:, 1] * heading[:, 0])
        half = W / 2 * np.sqrt(1 - tilt_sin ** 2) + L / 2 * tilt_sin

        # Nearby pairs only: j matters to i if it can be within i's look-ahead (or beside it).
        horizon = np.maximum(30.0, 4 * v + 10)
        diff = xy[None, :, :] - xy[:, None, :]
        dist2 = diff[..., 0] ** 2 + diff[..., 1] ** 2
        near = dist2 < (horizon + 7.0)[:, None] ** 2
        np.fill_diagonal(near, True)          # a self pair keeps every vehicle's segment non-empty
        I, J = np.nonzero(near)               # sorted by I
        starts = np.searchsorted(I, np.arange(n))

        # Each j seen from i, in i's route coordinates.
        S, D, ok, idx = net.project(r[I], xy[J])
        ok &= I != J
        t_ij = net.T[r[I], idx]
        heading_dot = np.sum(heading[J] * t_ij, axis=-1)
        cos = np.abs(heading_dot)
        sin = np.sqrt(np.maximum(0, 1 - cos ** 2))
        long_ext = cos * L[J] / 2 + sin * W[J] / 2
        lat_ext = sin * L[J] / 2 + cos * W[J] / 2
        v_long = np.sum(vel[J] * t_ij, axis=-1)
        dS = S - s[I]
        gap = dS - L[I] / 2 - long_ext
        # On the inside of a bend, distance along the centreline overstates the real gap. Close by, the
        # gap measured straight ahead of the follower is used when it is smaller (it never adds blockers:
        # whether j is in the path is still decided in route coordinates).
        rel = xy[J] - xy[I]
        lon = np.sum(rel * tan[I], axis=1)
        lat = np.sum(rel * nrm[I], axis=1)
        close_by = (lon > 0) & (lon < NEAR_FRAME_M) & (np.abs(lat) < half[I] + lat_ext + 0.5)
        gap = np.where(close_by & ok, np.minimum(gap, lon - L[I] / 2 - long_ext), gap)
        ahead = ok & (dS > 0) & (gap < horizon[I])

        signal_gap = signal_state = None
        if self.signals is not None:
            group = self.sig_group[r]
            controlled = group >= 0
            signal_state = np.full(n, -1, np.int64)
            if controlled.any():
                states = self.signals.states(self.t)
                signal_state = np.where(controlled, states[np.maximum(group, 0)], -1)
            signal_gap = self.sig_stop[r] - s - L / 2

        return View(t=self.t, dt=self.dt, net=net, slots=A, uid=self.uid[A], route=r, s=s, d=d, v=v, L=L, W=W,
                    xy=xy, heading=heading, vel=vel, tangent=tan, normal=nrm, half=half, horizon=horizon,
                    I=I, J=J, starts=starts, S=S, D=D, ok=ok, dS=dS, gap=gap, long_ext=long_ext, lat_ext=lat_ext,
                    v_long=v_long, heading_dot=heading_dot, ahead=ahead,
                    near_wide=dist2 < (horizon[:, None] + horizon[None, :] + 15.0) ** 2,
                    signal_gap=signal_gap, signal_state=signal_state)

    # ------------------------------------------------------------ one step

    def step(self):
        self._spawn()
        A = np.flatnonzero(self.active)
        if len(A) == 0:
            self._advance_clock()
            return
        dt = self.dt
        view = self.view(A)
        dec = self.model.decide(view)
        s, d, v = view.s, view.d, view.v

        # Integrate exactly what the drivers decided; nothing stops them moving into each other.
        v_new = np.maximum(v + dec.acc * dt, 0.0)
        self.s[A] = s + (v + v_new) / 2 * dt
        self.d[A] = d + dec.vd * dt
        self.v[A], self.vd[A], self.acc[A] = v_new, dec.vd, dec.acc
        self.model.after_move(view, v_new)
        self.stuck[A] = np.where(v_new < 0.1, self.stuck[A] + dt, 0.0)
        self.max_stuck = max(self.max_stuck, float(self.stuck[A].max()))

        if self.check_collisions:
            self._collisions(A, view.xy, view.heading, view.L, view.W)
        if self.steps % self.record_every == 0:
            self._record(view, dec)
        self._exits(A)
        self._advance_clock()

    def _advance_clock(self):
        self.steps += 1
        self.t = self.steps * self.dt

    # ------------------------------------------------------------ bookkeeping

    def _exits(self, A):
        done = self.s[A] >= self.net.length[self.route[A]] + self.L[A] / 2
        for k in A[done]:
            record = self.vehicles[int(self.uid[k])]
            record["t_exit"] = self.t + self.dt
            self.active[k] = False
            self.model.leave(k)

    def _record(self, view, dec):
        """One sample per vehicle: its state at this step's time (before the move) and what it decided."""
        self.samples.append(np.column_stack([view.uid, np.full(len(view.uid), self.t), view.xy, view.s, view.d,
                                             view.v, dec.acc, view.heading]))
        if self.signals is not None:
            self.signal_log.append((self.t, self.signals.states(self.t)))

    def _collisions(self, A, xy, heading, L, W):
        """Record every overlap of bodies (shrunk by CONTACT_SHRINK so a graze is not a contact):
        oriented boxes, separating-axis test on close pairs. Per pair of vehicles: when they first
        touched, how many separate contacts, total time overlapping, deepest overlap."""
        diff = xy[None, :, :] - xy[:, None, :]
        dist = np.linalg.norm(diff, axis=-1)
        reach = (np.hypot(L, W)[:, None] + np.hypot(L, W)[None, :]) / 2
        I, J = np.nonzero(np.triu(dist < reach, 1))
        if len(I) == 0:
            return
        hx = heading
        hy = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
        half = (1 - CONTACT_SHRINK) / 2
        depth = np.full(len(I), np.inf)       # overlap along the axis that separates them least
        for axis in (hx[I], hy[I], hx[J], hy[J]):
            proj = np.abs(np.sum(diff[I, J] * axis, axis=1))
            ri = half * (L[I] * np.abs(np.sum(hx[I] * axis, 1)) + W[I] * np.abs(np.sum(hy[I] * axis, 1)))
            rj = half * (L[J] * np.abs(np.sum(hx[J] * axis, 1)) + W[J] * np.abs(np.sum(hy[J] * axis, 1)))
            depth = np.minimum(depth, ri + rj - proj)
        touching = depth >= 0
        for i, j, dep in zip(I[touching], J[touching], depth[touching]):
            key = tuple(sorted((int(self.uid[A[i]]), int(self.uid[A[j]]))))
            rec = self.contacts.get(key)
            if rec is None:
                rec = self.contacts[key] = {"t_first": round(self.t, 2), "events": 1, "duration_s": 0.0,
                                            "max_depth_m": 0.0, "_last": self.steps}
            elif rec["_last"] < self.steps - 1:
                rec["events"] += 1
            rec["_last"] = self.steps
            rec["duration_s"] = round(rec["duration_s"] + self.dt, 3)
            rec["max_depth_m"] = round(max(rec["max_depth_m"], float(dep)), 3)
            if key not in self.collisions and self.debug:
                gi, gj = A[i], A[j]
                self.collision_log.append(dict(t=round(self.t, 2), uids=key,
                    a=(int(self.route[gi]), round(float(self.s[gi]), 1), round(float(self.d[gi]), 2), round(float(self.v[gi]), 2), int(self.kind[gi])),
                    b=(int(self.route[gj]), round(float(self.s[gj]), 1), round(float(self.d[gj]), 2), round(float(self.v[gj]), 2), int(self.kind[gj])),
                    cos=round(float(np.dot(heading[i], heading[j])), 2)))
            self.collisions.setdefault(key, round(self.t, 2))

    # ------------------------------------------------------------ running

    def run(self, duration, until_empty=False):
        end_step = int(round(duration / self.dt))
        while self.steps < end_step:
            self.step()
        if until_empty:
            limit = end_step * 4 + 6000
            while (self.active.any() or self.next_pending < len(self.pending)) and self.steps < limit:
                self.step()
        return self.result()

    def result(self):
        from .output import Result
        samples = np.concatenate(self.samples) if self.samples else np.zeros((0, 10))
        return Result(self.net, list(self.vehicles.values()), samples, dict(self.collisions), self.t,
                      unspawned=len(self.pending) - self.next_pending, max_stuck=self.max_stuck,
                      signal_log=self.signal_log, dt=self.dt * self.record_every,
                      contacts={k: {f: x for f, x in c.items() if not f.startswith("_")} for k, c in self.contacts.items()})
