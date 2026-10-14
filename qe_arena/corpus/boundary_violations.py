"""Build request values that each violate one schema boundary. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/functional_negative_agent.py (MIT)."""

META = {
    "entry": "boundary_violations",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/functional_negative_agent.py",
    "tags": ["api", "negative-testing", "boundary-values"],
    "domain": "software-engineering",
}


def boundary_violations(properties):
    """Return up to three property values that each break one JSON-schema boundary.

    properties maps a name to a schema dict and is visited in insertion order.
    - "integer" or "number": maximum + 1 if "maximum" is set, else minimum - 1 if
      "minimum" is set.
    - "string" (also the default when "type" is missing): "x" * (maxLength + 1) if
      "maxLength" is set, else "x" * (minLength - 1) if minLength is set and > 0.
    - "array": ["item"] * (maxItems + 1) if "maxItems" is set, else
      ["item"] * (minItems - 1) if minItems is set and > 0.
    Other types, and properties with no usable boundary, are skipped. Stops as soon
    as three modifications exist. Returns a dict of property name -> violating value.
    """
    modifications = {}
    for name, schema in properties.items():
        kind = schema.get("type", "string")
        if kind in ("integer", "number"):
            if "maximum" in schema:
                modifications[name] = schema["maximum"] + 1
            elif "minimum" in schema:
                modifications[name] = schema["minimum"] - 1
        elif kind == "string":
            if "maxLength" in schema:
                modifications[name] = "x" * (schema["maxLength"] + 1)
            elif "minLength" in schema and schema["minLength"] > 0:
                modifications[name] = "x" * (schema["minLength"] - 1)
        elif kind == "array":
            if "maxItems" in schema:
                modifications[name] = ["item"] * (schema["maxItems"] + 1)
            elif "minItems" in schema and schema["minItems"] > 0:
                modifications[name] = ["item"] * (schema["minItems"] - 1)
        if len(modifications) >= 3:
            break
    return modifications


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def _schema(rng):
        kind = rng.choice(["integer", "number", "string", "array", "boolean", None])
        schema = {} if kind is None else {"type": kind}
        names = {"integer": ("minimum", "maximum"), "number": ("minimum", "maximum"),
                 "string": ("minLength", "maxLength"), "array": ("minItems", "maxItems")}
        low_key, high_key = names.get(kind or "string", ("minLength", "maxLength"))
        if rng.random() < 0.6:
            schema[low_key] = rng.randint(0, 4)
        if rng.random() < 0.5:
            schema[high_key] = rng.randint(1, 6)
        return schema

    cases = [
        [{}],
        [{"a": {"type": "integer", "maximum": 10}}],
        [{"a": {"type": "number", "minimum": 0}}],
        [{"a": {"type": "integer", "minimum": 2, "maximum": 5}}],
        [{"a": {"type": "integer"}}],
        [{"s": {"maxLength": 3}}],
        [{"s": {"type": "string", "minLength": 0}}],
        [{"s": {"type": "string", "minLength": 1}}],
        [{"s": {"type": "string", "minLength": 3, "maxLength": 4}}],
        [{"s": {"type": "string", "minLength": 4}}],
        [{"l": {"type": "array", "maxItems": 1}}],
        [{"l": {"type": "array", "minItems": 1}}],
        [{"l": {"type": "array", "minItems": 3}}],
        [{"l": {"type": "array", "minItems": 0}}],
        [{"b": {"type": "boolean", "maximum": 1}}],
        [{"a": {"type": "integer", "maximum": 1}, "b": {"maxLength": 0}, "c": {"type": "array", "maxItems": 0}}],
        [{"a": {"type": "integer", "maximum": 1}, "b": {"maxLength": 0}, "c": {"type": "array", "maxItems": 0}, "d": {"type": "number", "minimum": 1}}],
        [{"a": {"type": "integer"}, "b": {"maxLength": 1}, "c": {"type": "boolean"}, "d": {"type": "number", "minimum": -3}, "e": {"type": "array", "minItems": 2}}],
    ]
    while len(cases) < n:
        cases.append([{f"p{i}": _schema(rng) for i in range(rng.randint(1, 5))}])
    return cases[:n]
