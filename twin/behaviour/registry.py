"""The behaviour models the tools know by name.

The search, the sensitivity analysis and twin.run never name a model class: they ask here. A model
becomes available in one of two ways:

  registered   @register("name") on the class, in a module that twin/behaviour/__init__.py imports
  by path      --model package.module:ClassName, for a class kept outside twin/ (the module must be
               importable from where the tool is started). Nothing has to be registered.

A parameter file stores the name it was made for (ParamSet.model), so make_model(ParamSet.load(file))
gives back the right model, in worker processes too.
"""
from __future__ import annotations

import importlib

_MODELS = {}


def register(name):
    """Class decorator: make a BehaviourModel available as --model NAME."""
    def wrap(cls):
        if _MODELS.get(name, cls) is not cls:
            raise ValueError(f"Behaviour model name {name!r} is already taken by {_MODELS[name].__name__}")
        cls.name = name
        _MODELS[name] = cls
        return cls
    return wrap


def names():
    return tuple(sorted(_MODELS))


def get(name):
    """The model class for a registered name or a module:ClassName path."""
    if name in _MODELS:
        return _MODELS[name]
    module, _, attribute = str(name).partition(":")
    if not attribute:
        raise KeyError(f"Unknown behaviour model {name!r}: registered are {', '.join(names())}; "
                       f"a model outside twin/ is named package.module:ClassName")
    try:
        cls = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise KeyError(f"Cannot load behaviour model {name!r}: {error}") from error
    _MODELS[name] = cls
    return cls


def default_params(name, scene=None):
    """The model's starting parameters for a scene (base.Scene; None: nothing is known about it)."""
    params = get(name).default_params(scene)
    params.model = name
    return params


def make_model(params):
    """A new instance of the behaviour model a parameter set belongs to."""
    return get(params.model)(params)
