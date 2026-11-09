from __future__ import annotations

import types


class _FlagValues(types.SimpleNamespace):
    def FlagValuesDict(self):
        return dict(vars(self))

    def __call__(self, argv):
        raise RuntimeError("gflags_shim does not parse command lines (POIBench adapter passes a namespace)")


FLAGS = _FlagValues()


def _define(name, default, *args, **kwargs):
    setattr(FLAGS, name, default)


DEFINE_string = DEFINE_integer = DEFINE_float = DEFINE_bool = DEFINE_boolean = _define


def DEFINE_enum(name, default, enum_values, *args, **kwargs):
    if default is not None and default not in enum_values:
        raise ValueError(f"default {default} of {name} not in {enum_values}")
    setattr(FLAGS, name, default)


def defaults() -> dict:
    return dict(vars(FLAGS))
