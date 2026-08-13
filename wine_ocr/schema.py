"""Convert a Pydantic model into a schema the structured-outputs compiler accepts.

The compiler supports a deliberate subset of JSON Schema. Pydantic emits several
keywords outside it (numeric bounds, string lengths, ``default``), and requires
two things the compiler insists on but Pydantic omits: ``additionalProperties:
false`` on every object and *all* properties listed in ``required``.

``$ref``/``$defs``, ``anyOf``, ``allOf``, ``enum`` and ``const`` are supported and
pass through untouched.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

# Dropped: unsupported by the schema compiler, or pure noise that costs tokens.
_STRIP = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "default", "examples", "title", "$schema",
}

_ALLOWED_FORMATS = {
    "date-time", "time", "date", "duration", "email", "hostname",
    "uri", "ipv4", "ipv6", "uuid",
}


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(n) for n in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRIP:
            continue
        if key == "format" and value not in _ALLOWED_FORMATS:
            continue
        out[key] = _clean(value)

    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        # Every property must be required; nullability is expressed in the type.
        out["required"] = list(out["properties"].keys())
    return out


def json_schema_for(model: Type[BaseModel]) -> dict:
    """Compiler-ready JSON schema for ``model``."""
    return _clean(model.model_json_schema())


def output_config(model: Type[BaseModel], effort: str | None = None) -> dict:
    """The ``output_config`` request field pinning responses to ``model``."""
    cfg: dict[str, Any] = {
        "format": {"type": "json_schema", "schema": json_schema_for(model)}
    }
    if effort:
        cfg["effort"] = effort
    return cfg
