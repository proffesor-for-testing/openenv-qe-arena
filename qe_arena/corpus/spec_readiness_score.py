"""Score how well an OpenAPI spec is documented for LLM test generation. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/spec_service/app_factory.py (MIT)."""

META = {
    "entry": "spec_readiness_score",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/spec_service/app_factory.py",
    "tags": ["api", "openapi", "scoring"],
    "domain": "software-engineering",
}


def spec_readiness_score(spec):
    """Return an integer readiness score from 0 to 100 for an OpenAPI-like dict.

    Points: "info" present +10, and +5 more if spec["info"] has "description".
    "paths" present +20. For each operation under spec["paths"][path][method] whose
    method is get/post/put/delete/patch (other methods are ignored): +1 if it has
    "parameters" and +1 if it has "responses". An operation is documented if it has
    "summary" or "description"; the documentation bonus is documented * 30 // total
    (integer division; no bonus when there are no counted operations).
    +20 if spec["components"] has "schemas", otherwise +20 if "definitions" is present.
    +15 if "security" or "securityDefinitions" is present. The total is capped at 100.
    """
    score = 0
    if "info" in spec:
        score += 10
        if "description" in spec["info"]:
            score += 5
    if "paths" in spec:
        score += 20
        total = documented = 0
        for methods in spec["paths"].values():
            for method, operation in methods.items():
                if method in ("get", "post", "put", "delete", "patch"):
                    total += 1
                    if "summary" in operation or "description" in operation:
                        documented += 1
                    if "parameters" in operation:
                        score += 1
                    if "responses" in operation:
                        score += 1
        if total > 0:
            score += documented * 30 // total
    if "components" in spec and "schemas" in spec["components"]:
        score += 20
    elif "definitions" in spec:
        score += 20
    if "security" in spec or "securityDefinitions" in spec:
        score += 15
    return min(score, 100)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def _operation(rng):
        keys = ["summary", "description", "parameters", "responses"]
        return {k: 1 for k in keys if rng.random() < 0.5}

    doc = {"summary": 1, "responses": 1}
    bare = {"responses": 1}
    cases = [
        [{}],
        [{"info": {}}],
        [{"info": {"description": "d"}}],
        [{"paths": {}}],
        [{"paths": {"/a": {"head": doc, "options": doc}}}],
        [{"paths": {"/a": {"get": {"description": 1}}}}],
        [{"paths": {"/a": {"get": doc, "post": bare, "put": bare}}}],
        [{"paths": {"/a": {"get": doc, "post": doc, "put": bare}}}],
        [{"paths": {"/a": {"get": doc, "delete": bare}, "/b": {"patch": {"parameters": 1}, "post": bare}}}],
        [{"paths": {"/a": {"get": doc, "post": doc, "put": doc}, "/b": {"patch": bare, "delete": bare}, "/c": {"get": bare, "post": bare}}}],
        [{"components": {}}],
        [{"components": {"schemas": {}}}],
        [{"components": {}, "definitions": {}}],
        [{"definitions": {}}],
        [{"security": []}],
        [{"securityDefinitions": {}}],
        [{"info": {"description": "d"}, "paths": {"/a": {m: {"summary": 1, "parameters": 1, "responses": 1} for m in ("get", "post", "put", "delete", "patch")}}, "components": {"schemas": {}}, "security": []}],
        [{"info": {"description": "d"}, "paths": {"/a": {"get": doc}}, "components": {"schemas": {}}, "security": []}],
    ]
    while len(cases) < n:
        spec = {}
        if rng.random() < 0.7:
            spec["info"] = {"description": "d"} if rng.random() < 0.5 else {}
        if rng.random() < 0.8:
            methods = ["get", "post", "put", "delete", "patch", "head"]
            spec["paths"] = {
                f"/r{i}": {m: _operation(rng) for m in rng.sample(methods, rng.randint(1, 3))}
                for i in range(rng.randint(0, 3))
            }
        if rng.random() < 0.5:
            spec["components"] = {"schemas": {}} if rng.random() < 0.6 else {}
        if rng.random() < 0.3:
            spec["definitions"] = {}
        if rng.random() < 0.5:
            spec[rng.choice(["security", "securityDefinitions"])] = []
        cases.append([spec])
    return cases[:n]
