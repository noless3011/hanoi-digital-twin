"""A model's parameters, declared once so an optimiser can treat them as a vector.

Each parameter (Spec) has a default, bounds, a scale (log or linear) and a stage (the order in which it
is worth fitting). A ParamSet holds values for one model's list of Specs and the model's name. A model may
keep per-kind ratios in it: values are then the motorcycle's, and another kind's value is the
motorcycle's times a fixed ratio, so the search space stays small.

SPECS and RATIOS below are the choice model's (choice.py); ParamSet() without arguments is its defaults.

    p = ParamSet().with_values({"v0": 7.5})
    x = p.to_unit(names)            # each in 0..1 on the parameter's own scale
    p = p.from_unit(names, x)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    name: str
    default: float
    lo: float
    hi: float
    log: bool
    stage: int
    doc: str


SPECS = (
    Spec("v0", 8.0, 3.0, 14.0, True, 1, "desired speed on a free road (m/s)"),
    Spec("v0_spread", 0.15, 0.02, 0.5, True, 1, "spread of desired speed between drivers (log-normal sigma)"),
    Spec("v0_seen", 1.1, 0.7, 1.6, True, 1, "desired speed as a multiple of the speed a vehicle was seen at on "
                                           "video (Arrival.v_seen); v0 and v0_spread are for vehicles without one"),
    Spec("a", 1.8, 0.5, 4.0, True, 1, "acceleration when pulling away (m/s2)"),
    Spec("b", 2.5, 1.0, 5.0, True, 2, "comfortable braking (m/s2)"),
    Spec("T", 0.8, 0.1, 2.5, True, 2, "time gap wanted to whatever is ahead (s)"),
    Spec("s0", 0.8, 0.2, 3.0, True, 2, "gap wanted at standstill (m)"),
    Spec("clat", 0.3, 0.05, 1.5, True, 3, "sideways clearance wanted when passing (m)"),
    Spec("vlat", 0.8, 0.2, 2.0, True, 3, "fastest sideways drift (m/s)"),
    Spec("w_pos", 0.05, 0.005, 2.0, True, 3, "pull back to the driver's own position across the road (cost per m2)"),
    Spec("tau", 2.0, 0.8, 3.5, False, 4, "how far ahead the driver predicts (s)"),
    Spec("w_risk", 3.0, 0.3, 30.0, True, 4, "weight of closeness to others against making progress"),
    Spec("T_imp", 15.0, 3.0, 60.0, True, 4, "waiting time that doubles the urge to make progress (s)"),
)
BY_NAME = {s.name: s for s in SPECS}
NAMES = tuple(s.name for s in SPECS)

# Other kinds relative to the motorcycle (parameters not listed: 1).
RATIOS = {
    "motorcycle": {},
    "bicycle": {"v0": 0.5, "a": 0.6, "vlat": 0.7},
    "car": {"v0": 0.95, "a": 0.8, "T": 1.4, "s0": 2.0, "clat": 1.6, "vlat": 0.6},
    "van": {"v0": 0.9, "a": 0.7, "T": 1.5, "s0": 2.0, "clat": 1.6, "vlat": 0.55},
    "truck": {"v0": 0.8, "a": 0.55, "T": 1.8, "s0": 2.5, "clat": 2.0, "vlat": 0.45},
    "bus": {"v0": 0.8, "a": 0.55, "T": 1.8, "s0": 2.5, "clat": 2.0, "vlat": 0.45},
}


def turning_specs(turns):
    """Parameters for `turns` turning-share shifts (Scene.turns). They belong to the demand, not to a
    driver: a model that adds them to its parameter list has them fitted with its own (the search and
    twin.run re-draw the ambiguous arrivals under them, measure/demand.py), and never reads them itself."""
    return [Spec(f"turn_{k}", 0.0, -2.5, 2.5, False, 0,
                 f"demand: log-odds shift of the main entry's route {k} against its first route")
            for k in range(1, turns + 1)]


def stage_names(max_stage):
    return tuple(s.name for s in SPECS if s.stage <= max_stage)


class ParamSet:
    """Values for a list of Specs. Without `specs` these are the choice model's (SPECS and RATIOS above);
    another model passes its own specs and, if it wants them, its own ratios ({kind: {name: ratio}})."""

    def __init__(self, values=None, ratios=None, specs=None, model="choice"):
        self.specs = tuple(specs) if specs is not None else SPECS
        self.model = model
        base_ratios = RATIOS if specs is None else {}
        self.by_name = {s.name: s for s in self.specs}
        unknown = set(values or {}) - set(self.by_name)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
        self.values = {s.name: float((values or {}).get(s.name, s.default)) for s in self.specs}
        self.ratios = {k: {**base_ratios.get(k, {}), **(ratios or {}).get(k, {})}
                       for k in {**base_ratios, **(ratios or {})}}

    @property
    def names(self):
        return tuple(s.name for s in self.specs)

    def of(self, kind):
        """The values of one vehicle kind."""
        r = self.ratios.get(kind, {})
        return {name: value * r.get(name, 1.0) for name, value in self.values.items()}

    def with_values(self, values):
        return ParamSet({**self.values, **values}, self.ratios, self.specs, self.model)

    # ---- the optimiser's view: each parameter in 0..1 on its own scale

    def to_unit(self, names):
        out = []
        for name in names:
            s, v = self.by_name[name], self.values[name]
            out.append((math.log(v / s.lo) / math.log(s.hi / s.lo)) if s.log else (v - s.lo) / (s.hi - s.lo))
        return out

    def from_unit(self, names, x):
        values = {}
        for name, u in zip(names, x):
            s = self.by_name[name]
            u = min(max(float(u), 0.0), 1.0)
            values[name] = s.lo * (s.hi / s.lo) ** u if s.log else s.lo + u * (s.hi - s.lo)
        return self.with_values(values)

    # ---- files

    def to_json(self):
        out = {"schema": "twin-params", "version": 2, "model": self.model, "values": self.values}
        if any(self.ratios.values()):
            out["ratios"] = self.ratios
        if self.specs != SPECS:
            out["specs"] = [[s.name, s.default, s.lo, s.hi, s.log, s.stage, s.doc] for s in self.specs]
        return out

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") not in ("twin-params", "twin-choice-params"):
            raise ValueError("Not a twin parameter file")
        specs = [Spec(*row) for row in data["specs"]] if data.get("specs") else None
        return cls(data["values"], data.get("ratios"), specs, data.get("model", "choice"))
