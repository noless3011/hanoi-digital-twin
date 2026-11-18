"""Video demand: which vehicles entered, when, on which route, so the simulation replays the same traffic.

A (stitched) track that is first seen near the start of the visible part of its route is an arrival.
Its time at the route start is extrapolated back from its first sighting. Tracks first seen further in
are tracker re-acquisitions that stitching could not join; they are counted, not replayed.

A track that disappears before its route splits has several candidate routes; it gets one drawn from
the turning shares of tracks from the same entry that were seen past the split (seeded, so the same
video always gives the same demand).
"""
from __future__ import annotations

import numpy as np

from ..engine import Arrival

ENTRY_TOLERANCE_M = 15.0
SEEN_PERCENTILE = 85        # a track's free speed: this percentile of its smoothed speed
MIN_SEEN_S = 1.0            # tracks shorter than this give no free speed
MIN_SEEN_SPEED = 2.0        # nor do tracks never faster than this (m/s): queued throughout, it says nothing


def seen_speed(tr):
    """The speed a track shows when it is not held up: SEEN_PERCENTILE of its smoothed speed, or None when
    the track is too short or too slow to show it. A behaviour model may take it as the driver's desired
    speed (choice model: v0_seen)."""
    if tr.duration < MIN_SEEN_S:
        return None
    v = float(np.percentile(tr.speed, SEEN_PERCENTILE))
    return v if v >= MIN_SEEN_SPEED else None


def _visible_start(network, r):
    inside = network.in_view(network.routes[r].xy)
    k = np.flatnonzero(inside)
    return float(network.routes[r].s[k[0]]) if len(k) else 0.0


def from_tracks(tracks, network, *, seed=0):
    """Returns (arrivals, report)."""
    rng = np.random.default_rng(seed)
    starts = {r: _visible_start(network, r) for r in range(len(network.routes))}
    # Turning shares per entry lane from tracks with a single candidate.
    shares = {}
    for tr in tracks:
        if len(tr.candidates) == 1:
            entry = network.routes[tr.route].lanes[0]
            shares.setdefault(entry, {}).setdefault(tr.route, 0)
            shares[entry][tr.route] += 1
    arrivals, late, unmatched = [], 0, 0
    for tr in tracks:
        if tr.route < 0:
            unmatched += 1
            continue
        if tr.s[0] > starts[tr.route] + ENTRY_TOLERANCE_M:
            late += 1
            continue
        route = tr.route
        if len(tr.candidates) > 1:
            entry = network.routes[route].lanes[0]
            known = {r: n for r, n in shares.get(entry, {}).items() if r in tr.candidates}
            if known:
                routes, counts = list(known), np.array(list(known.values()), float)
                route = int(routes[rng.choice(len(routes), p=counts / counts.sum())])
        v_first = float(np.median(tr.speed[: min(5, len(tr.speed))]))
        t0 = float(tr.t[0] - max(tr.s[0], 0) / max(v_first, 1.0))
        right, left = network.DR[route, 0], network.DL[route, 0]
        d = float(np.clip(tr.d[0], right, left))
        arrivals.append(Arrival(t0, tr.kind, route, uid=tr.id, d=d, v=v_first, v_seen=seen_speed(tr)))
    arrivals.sort()
    t_min = min((a.t for a in arrivals), default=0.0)
    if t_min < 0:   # vehicles already inside at the start of the clip
        for a in arrivals:
            a.t -= t_min
    report = {"tracks": len(tracks), "arrivals": len(arrivals),
              "with_seen_speed": sum(a.v_seen is not None for a in arrivals), "first_seen_inside": late,
              "off_layout": unmatched, "time_shift_s": round(-min(t_min, 0.0), 2),
              "by_route": {network.routes[r].id: sum(a.route == r for a in arrivals) for r in range(len(network.routes))}}
    return arrivals, report


# ------------------------------------------------------------------ turning shares as parameters

def turning_setup(tracks, network, arrivals, *, seed=0):
    """What is needed to re-draw the routes of arrivals whose track fits several routes, for the entry
    with the most such arrivals (the main entry). Returns None when there is nothing to re-draw, else
      routes   that entry's route indices, in order (turning parameter k shifts routes[k] against routes[0])
      prior    the share of each of those routes among arrivals with a single possible route
      options  {uid: {"u": a fixed number in 0..1, "routes": {route: (entry time, entry offset)}}}
    The fixed number makes the re-draw a smooth function of the shares: the same vehicle changes route
    only when a share boundary passes its number."""
    by_id = {tr.id: tr for tr in tracks}
    rng = np.random.default_rng([seed, 7])
    options, per_entry = {}, {}
    for a in arrivals:
        tr = by_id.get(a.uid)
        u = float(rng.random())          # one draw per arrival, whether or not it is used
        if tr is None or len(tr.candidates) < 2:
            continue
        speed = max(a.v if a.v is not None else 1.0, 1.0)
        shift = a.t - (tr.t[0] - max(tr.s[0], 0) / speed)
        routes = {}
        for r in tr.candidates:
            st, d, ok, _ = network.project(np.array(r), tr.xy[0])
            if ok:
                routes[int(r)] = (max(float(tr.t[0] - max(float(st), 0) / speed + shift), 0.0),
                                  float(np.clip(d, network.DR[r, 0], network.DL[r, 0])))
        if len(routes) > 1:
            options[a.uid] = {"u": u, "routes": routes}
            entry = network.routes[a.route].lanes[0]
            per_entry[entry] = per_entry.get(entry, 0) + 1
    if not per_entry:
        return None
    entry = max(per_entry, key=per_entry.get)
    routes = [r.index for r in network.routes if r.lanes[0] == entry]
    counts = np.array([1.0 + sum(a.route == r and a.uid not in options for a in arrivals) for r in routes])
    keep = {uid: o for uid, o in options.items() if set(o["routes"]) <= set(routes)}
    return {"routes": routes, "prior": (counts / counts.sum()).tolist(), "options": keep}


def turning_shifts(values, turning):
    """The turn_<k> values of a parameter dict (behaviour/params.py turning_specs) as a list, one per
    route after the first."""
    return [float(values.get(f"turn_{k}", 0.0)) for k in range(1, len(turning["routes"]))]


def apply_turning(arrivals, turning, shifts):
    """Arrivals with the main entry's ambiguous vehicles re-drawn under shifted turning shares.
    shifts: log-odds of routes[1:] against routes[0], added to the prior's."""
    routes = turning["routes"]
    weight = np.array(turning["prior"]) * np.exp(np.concatenate([[0.0], np.asarray(shifts, float)]))
    share = dict(zip(routes, weight / weight.sum()))
    out = []
    for a in arrivals:
        option = turning["options"].get(a.uid)
        if option is None:
            out.append(Arrival(a.t, a.kind, a.route, a.uid, a.d, a.v, a.v0, a.v_seen))
            continue
        possible = sorted(option["routes"])
        edges = np.cumsum([share[r] for r in possible])
        route = possible[min(int(np.searchsorted(edges, option["u"] * edges[-1], "right")), len(possible) - 1)]
        t, d = option["routes"][route]
        out.append(Arrival(t, a.kind, route, a.uid, d, a.v, a.v0, a.v_seen))
    out.sort()
    return out
