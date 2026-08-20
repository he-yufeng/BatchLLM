"""A small JSON-schema subset validator for row outputs.

Deliberately dependency-free: this is not a full JSON Schema implementation,
just the parts that matter for validating an LLM row before it feeds the next
step — type, required/properties nesting, items, and enum. Anything outside
the subset is ignored rather than rejected, so a schema that uses richer
keywords still validates against the parts BatchLLM understands.
"""

from __future__ import annotations

from typing import Any


def _type_matches(data: Any, expected: str) -> bool:
    # bool is a subclass of int in Python; check it first so true/false never
    # satisfy a numeric constraint.
    if expected == "boolean":
        return isinstance(data, bool)
    if expected == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if expected == "number":
        return isinstance(data, (int, float)) and not isinstance(data, bool)
    if expected == "string":
        return isinstance(data, str)
    if expected == "object":
        return isinstance(data, dict)
    if expected == "array":
        return isinstance(data, list)
    if expected == "null":
        return data is None
    return True  # unknown type name: ignore rather than reject


def validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> str | None:
    """Return a human-readable violation, or None when the data satisfies the schema."""
    if not isinstance(schema, dict):
        return None

    if "enum" in schema and isinstance(schema["enum"], list):
        if not any(data == value and type(data) is type(value) for value in schema["enum"]):
            return f"{path}: {data!r} not in enum {schema['enum']}"

    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_type_matches(data, t) for t in types):
            return f"{path}: expected type {'/'.join(types)}, got {type(data).__name__}"

    if isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                return f"{path}.{req}: missing required property"
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in data and isinstance(subschema, dict):
                    violation = validate_schema(data[key], subschema, f"{path}.{key}")
                    if violation is not None:
                        return violation

    if isinstance(data, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(data):
                violation = validate_schema(item, items, f"{path}[{i}]")
                if violation is not None:
                    return violation

    return None
