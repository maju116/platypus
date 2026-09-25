"""The R-facing edge of the engine.

Exceptions do not survive the crossing usefully: reticulate hands R the message as text
and drops the object, so the structured problems pyplatypus reports would be lost and the
R side would be left parsing its own prose. Everything here therefore returns data.

This lives in the R package rather than in pyplatypus because it exists for R's benefit.
pyplatypus raises exceptions, like any Python library should.
"""

from __future__ import annotations

from typing import Any


def _failure(error: Any) -> dict:
    payload = error.to_dict() if hasattr(error, "to_dict") else {}
    payload.setdefault("kind", type(error).__name__)
    payload.setdefault("message", str(error))
    payload.setdefault("problems", [])
    payload["ok"] = False
    return payload


def build_spec(config: dict, check_paths: bool = True) -> dict:
    """A validated spec, or the reasons it is not one."""
    import pyplatypus

    try:
        return {"ok": True, "spec": pyplatypus.from_dict(config, check_paths=check_paths)}
    except pyplatypus.PlatypusError as error:
        return _failure(error)


def load_spec(path: str, check_paths: bool = True) -> dict:
    """The same, from a YAML file. The same object comes out either way."""
    import pyplatypus

    try:
        return {"ok": True, "spec": pyplatypus.from_yaml(path, check_paths=check_paths)}
    except pyplatypus.PlatypusError as error:
        return _failure(error)


def spec_as_dict(spec: Any) -> dict:
    """Plain data, so R can look at a spec without asking Python about every field."""
    return spec.to_dict()
