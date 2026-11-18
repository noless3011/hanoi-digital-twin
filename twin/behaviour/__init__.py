"""Behaviour models: how drivers decide. The engine (twin/engine.py) moves vehicles; a model decides.
The contract is in base.py, with how to add a model; registry.py is how the tools find one by name.

  constant_speed.py   follows its route at one speed, keeps its sideways offset, ignores everyone
  choice.py           compares a few actions each step and takes the cheapest (parameters in params.py)
  priority.py         reacts only to vehicles senior to it, so it cannot deadlock; speed profiles per route
"""
from .base import BehaviourModel, Decision, Scene, View
from .params import ParamSet, Spec
from .registry import default_params, make_model, register
from .registry import names as model_names
# Importing a model's module registers it.
from .choice import ChoiceModel, CoreChoiceModel
from .constant_speed import ConstantSpeedModel
from .priority import PriorityModel
from .flow import FlowModel

__all__ = ["BehaviourModel", "Decision", "Scene", "View", "ConstantSpeedModel", "ChoiceModel", "CoreChoiceModel", "FlowModel", "PriorityModel",
           "ParamSet", "Spec", "register", "make_model", "default_params", "model_names"]
