"""Simulation results: vehicle table, samples, summary, and export to the vehicle-trajectories JSON
that video tracking also writes (so the same measurement code and the Unity player read both)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .signals import NAMES as SIGNAL_NAMES

COLUMNS = ("uid", "t", "x", "y", "s", "d", "v", "acc", "hx", "hy")
C = {name: k for k, name in enumerate(COLUMNS)}


class Result:
    def __init__(self, network, vehicles, samples, collisions, end_t, *, unspawned=0, max_stuck=0.0, signal_log=(),
                 dt=0.1, contacts=None):
        self.network, self.vehicles, self.samples = network, vehicles, samples
        self.collisions, self.end_t, self.unspawned, self.max_stuck = collisions, end_t, unspawned, max_stuck
        self.signal_log, self.dt = list(signal_log), dt
        # (uid, uid) -> {"t_first", "events", "duration_s", "max_depth_m"}: every overlap of two bodies.
        self.contacts = dict(contacts or {})

    def column(self, name):
        return self.samples[:, C[name]]

    def tracks(self):
        """{uid: samples of that vehicle, time-ordered}."""
        order = np.lexsort((self.column("t"), self.column("uid")))
        rows = self.samples[order]
        uids, starts = np.unique(rows[:, 0], return_index=True)
        bounds = list(starts[1:]) + [len(rows)]
        return {int(u): rows[a:b] for u, a, b in zip(uids, starts, bounds)}

    def travel_times(self):
        """{route index: [entry-to-exit seconds of vehicles that finished]}."""
        out = {}
        for v in self.vehicles:
            if v["t_exit"] is not None:
                out.setdefault(v["route"], []).append(v["t_exit"] - v["t_spawn"])
        return out

    def summary(self):
        spawned = len(self.vehicles)
        exited = sum(v["t_exit"] is not None for v in self.vehicles)
        delays = [v["t_spawn"] - v["t_sched"] for v in self.vehicles]
        times = self.travel_times()
        return {"simulated_s": round(self.end_t, 2), "spawned": spawned, "exited": exited,
                "not_spawned": self.unspawned, "collisions": len(self.collisions),
                **self.contact_score(),
                "max_stopped_s": round(self.max_stuck, 1),
                "spawn_delay_mean_s": round(float(np.mean(delays)), 2) if delays else None,
                "spawn_delay_max_s": round(float(np.max(delays)), 2) if delays else None,
                "travel_time_median_s": {self.network.routes[r].id: round(float(np.median(t)), 2)
                                         for r, t in sorted(times.items())}}

    def contact_score(self):
        """How unsafe the run was: pairs that touched, separate contacts, total pair-seconds of overlap,
        and the deepest overlap (m)."""
        c = self.contacts.values()
        return {"contact_pairs": len(self.contacts), "contact_events": sum(x["events"] for x in c),
                "contact_time_s": round(sum(x["duration_s"] for x in c), 2),
                "max_contact_depth_m": round(max((x["max_depth_m"] for x in c), default=0.0), 3)}

    def exits_per_window(self, window_s):
        """Exits counted in consecutive windows (for throughput checks)."""
        t = np.array([v["t_exit"] for v in self.vehicles if v["t_exit"] is not None])
        bins = np.arange(0, self.end_t + window_s, window_s)
        return np.histogram(t, bins=bins)[0]

    def to_trajectories(self, feed=None, source="simulation"):
        """The vehicle-trajectories schema used by video export. world_position is [x, ground y, y]
        (Unity x, y, z) so a Unity layout network round-trips."""
        g = self.network.ground_y
        vehicles = []
        for uid, rows in self.tracks().items():
            meta = next(v for v in self.vehicles if v["uid"] == uid)
            observations = [{"time_s": round(float(r[C["t"]]), 3),
                             "world_position": [round(float(r[C["x"]]), 3), g, round(float(r[C["y"]]), 3)],
                             "speed_mps": round(float(r[C["v"]]), 3),
                             "heading": [round(float(r[C["hx"]]), 4), round(float(r[C["hy"]]), 4)],
                             "station_m": round(float(r[C["s"]]), 3), "offset_m": round(float(r[C["d"]]), 3)}
                            for r in rows]
            vehicles.append({"track_id": uid, "kind": meta["kind"], "route": self.network.routes[meta["route"]].id,
                             "termination": "exited" if meta["t_exit"] is not None else "end_of_run",
                             "observations": observations})
        out = {"schema": "vehicle-trajectories", "version": 1, "source": source, "feed": feed or self.network.name,
               "duration_s": round(self.end_t, 3), "fps": round(1 / self.dt, 3),
               "world_coordinate_system": "unity_xz_metres", "vehicles": vehicles}
        if self.signal_log:
            out["signals"] = [{"time_s": round(t, 3), "states": [SIGNAL_NAMES[s] for s in states]}
                              for t, states in self.signal_log[::max(1, int(round(0.5 / self.dt)))]]
        return out

    def save(self, path, feed=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_trajectories(feed)), encoding="utf-8")
