"""The contract between the simulation engine (twin/engine.py) and a behaviour model.

The engine owns the world: vehicle bodies and kinematics (route, s, d, v, sideways speed), spawning,
perception geometry, integration, exits, recording and contact checks. It enforces no driving rules:
vehicles move exactly as the model decides, through each other if the model lets them, and every
overlap of bodies is recorded as a contact. A behaviour model owns the
drivers: their parameters, whatever they remember between steps, and every
decision (how hard to accelerate, where to drift sideways, whom to give way to).

Each step the engine builds one View for all active vehicles at once (arrays, for NumPy speed), the
model returns one Decision, and the engine moves the vehicles. Vehicles are addressed by slot: a fixed
index into storage that a vehicle keeps while it is on the road. A model keeps its own per-slot arrays
and resizes them when the engine asks.

A model instance holds the state of one simulation: create a new one for each Simulation.

Adding a model (the tools then fit, analyse and run it like the others; see registry.py):

    @register("mine")
    class MyModel(BehaviourModel):
        SPECS = (Spec("v0", 8.0, 3.0, 14.0, True, 1, "desired speed (m/s)"), ...)   # what the search may tune

        def sample_driver(self, kind, arrival, rng, right, left, width): ...
        def decide(self, view): ...            # reads self.params.values["v0"]

What the search needs from a model: a short list of continuous, bounded parameters (SPECS, or
default_params() when the list depends on the scene); the same number of random draws from `rng` whatever
the parameters are, so that two runs with one seed differ only by their parameters; and a simulation of
a clip in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ParamSet


@dataclass
class Scene:
    """What is known about a place and its traffic when a model's parameter list is built. A model whose
    parameters do not depend on the place ignores it; every field may be missing."""
    network: object = None
    route_arrivals: np.ndarray | None = None   # arrivals per route in the clips being fitted
    turns: int = 0                             # turning shares that can be fitted (params.py turning_specs)


@dataclass
class View:
    """What drivers perceive this step. Per-vehicle arrays have one row per active vehicle (in slot
    order, `slots`); pair arrays have one row per nearby (i, j) pair, I and J indexing those rows."""
    t: float
    dt: float
    net: object
    slots: np.ndarray        # (n,) storage slots of the active vehicles
    uid: np.ndarray          # (n,)
    route: np.ndarray        # (n,)
    s: np.ndarray            # (n,) station along the route (m)
    d: np.ndarray            # (n,) sideways offset (m)
    v: np.ndarray            # (n,) speed along the route (m/s)
    L: np.ndarray            # (n,) body length (m)
    W: np.ndarray            # (n,) body width (m)
    xy: np.ndarray           # (n, 2) world position
    heading: np.ndarray      # (n, 2) body direction
    vel: np.ndarray          # (n, 2) world velocity
    tangent: np.ndarray      # (n, 2) route direction at the vehicle
    normal: np.ndarray       # (n, 2) route left normal at the vehicle
    half: np.ndarray         # (n,) sideways half extent of the body as tilted by its drift (m)
    horizon: np.ndarray      # (n,) look-ahead distance (m)
    # Nearby pairs, sorted by I; every vehicle has its self pair, so `starts` segments are never empty.
    I: np.ndarray
    J: np.ndarray
    starts: np.ndarray       # first pair row of each vehicle (for np.*.reduceat)
    S: np.ndarray            # j's position projected on i's route: station
    D: np.ndarray            #   and sideways offset
    ok: np.ndarray           # projection valid and i != j
    dS: np.ndarray           # S - s_i
    gap: np.ndarray          # bumper gap from i to j along i's route (straight-ahead gap when close by)
    long_ext: np.ndarray     # j's half extent along i's route
    lat_ext: np.ndarray      # j's half extent across i's route
    v_long: np.ndarray       # j's speed along i's route (negative when oncoming)
    heading_dot: np.ndarray  # cosine between j's heading and i's route direction
    ahead: np.ndarray        # j is ahead of i, within i's horizon
    near_wide: np.ndarray    # (n, n) pairs close enough to meet at a crossing point
    signal_gap: np.ndarray | None = None    # (n,) gap from the front to the stop line, inf if uncontrolled
    signal_state: np.ndarray | None = None  # (n,) 0 green, 1 amber, 2 red (None without signals)

    def red_min(self, x):
        """Per-vehicle minimum over its pair rows."""
        return np.minimum.reduceat(x, self.starts, axis=0)

    def red_max(self, x):
        return np.maximum.reduceat(x, self.starts, axis=0)


@dataclass
class Decision:
    """What drivers do this step, per active vehicle. The engine integrates exactly this and nothing
    else: it does not stop vehicles from moving into each other. Bodies may overlap; every overlap is
    recorded as a contact (Result.contacts), which is how a model's safety is measured."""
    acc: np.ndarray          # longitudinal acceleration (m/s2); speed never goes below zero
    vd: np.ndarray           # sideways speed (m/s)


class BehaviourModel:
    """Base class: override the hooks a model needs."""

    debug = False
    name = None      # set by registry.register
    SPECS = ()       # the parameters a search may tune (params.Spec), when they are the same everywhere

    def __init__(self, params=None):
        self.params = params if params is not None else self.default_params()

    @classmethod
    def default_params(cls, scene=None):
        """The starting ParamSet. Override when the parameter list depends on the scene (a Scene or None)."""
        return ParamSet(specs=cls.SPECS, model=cls.name)

    def bind(self, network, capacity):
        """Called once by the engine before the first vehicle."""
        self.net = network
        self.resize(capacity, 0)

    def resize(self, capacity, n_old):
        """Grow per-slot storage to `capacity`, keeping the first n_old slots."""

    def sample_driver(self, kind, arrival, rng, right, left, width):
        """A new driver for an arrival, entering now at the route start. right and left are the road
        edges there (signed offsets), width the body width. Returns a dict with at least "v" (speed at
        entry, m/s) and "d" (sideways offset at entry, m); the whole dict is passed to enter(). The
        engine places the vehicle there whether or not another one is in the way."""
        raise NotImplementedError

    def enter(self, slot, driver):
        """The driver got onto the road in this slot."""

    def leave(self, slot):
        """The vehicle in this slot left the road."""

    def decide(self, view: View) -> Decision:
        raise NotImplementedError

    def after_move(self, view: View, v_new):
        """Called after the engine moved the vehicles (v_new: their new speeds)."""
