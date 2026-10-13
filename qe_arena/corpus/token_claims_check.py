"""Validate the time and audience claims of a decoded bearer token. Original."""

META = {
    "entry": "token_claims_check",
    "origin": "Original.",
    "tags": ["security", "authentication", "tokens", "time-window"],
    "domain": "cybersecurity",
}


def token_claims_check(claims, now, audience, leeway):
    """Check a decoded token's time and audience claims (the signature was checked elsewhere).

    now and leeway are int seconds. Returns "ok" or the first failing reason in this order:
    "malformed": "exp" is missing, or "exp", "nbf" or "iat" is present but not an int (a bool
    is not an int); "expired": now >= exp + leeway; "not_yet_valid": "nbf" is present and
    now + leeway < nbf; "issued_in_future": "iat" is present and iat > now + leeway; "audience":
    "aud" is neither audience itself nor a list containing it. Raises ValueError if leeway < 0.
    """
    if leeway < 0:
        raise ValueError("leeway < 0")
    if "exp" not in claims:
        return "malformed"
    for key in ("exp", "nbf", "iat"):
        if key in claims and (isinstance(claims[key], bool) or not isinstance(claims[key], int)):
            return "malformed"
    if now >= claims["exp"] + leeway:
        return "expired"
    if "nbf" in claims and now + leeway < claims["nbf"]:
        return "not_yet_valid"
    if claims.get("iat", now) > now + leeway:
        return "issued_in_future"
    aud = claims.get("aud")
    accepted = [aud] if isinstance(aud, str) else aud if isinstance(aud, list) else []
    if audience not in accepted:
        return "audience"
    return "ok"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    api = "api"
    cases = [
        [{"exp": 200, "aud": api}, 100, api, 0], [{"aud": api}, 100, api, 0],
        [{"exp": "200", "aud": api}, 100, api, 0], [{"exp": True, "aud": api}, 100, api, 0],
        [{"exp": 200, "nbf": 1.5, "aud": api}, 100, api, 0], [{"exp": 200, "iat": None, "aud": api}, 100, api, 0],
        [{"exp": 100, "aud": api}, 100, api, 0], [{"exp": 100, "aud": api}, 100, api, 1],
        [{"exp": 99, "aud": api}, 100, api, 1], [{"exp": 200, "nbf": 101, "aud": api}, 100, api, 0],
        [{"exp": 200, "nbf": 101, "aud": api}, 100, api, 1], [{"exp": 200, "nbf": 100, "aud": api}, 100, api, 0],
        [{"exp": 200, "iat": 101, "aud": api}, 100, api, 0], [{"exp": 200, "iat": 101, "aud": api}, 100, api, 1],
        [{"exp": 200, "iat": 100, "aud": api}, 100, api, 0], [{"exp": 200, "aud": "web"}, 100, api, 0],
        [{"exp": 200, "aud": ["web", api]}, 100, api, 0], [{"exp": 200, "aud": ["web"]}, 100, api, 0],
        [{"exp": 200}, 100, api, 0], [{"exp": 200, "aud": {"api": 1}}, 100, api, 0],
        [{"exp": 200, "aud": "apis"}, 100, api, 0], [{"exp": 200, "aud": api}, 100, api, -1],
        [{"exp": 50, "nbf": 300, "aud": "x"}, 100, api, 0], [{"exp": 200, "nbf": 300, "iat": 300}, 100, api, 0],
        [{"exp": 200, "nbf": 0, "iat": 0, "aud": [api]}, 100, api, 0],
    ]
    while len(cases) < n:
        now = rng.randint(100, 199)
        claims = {"exp": now + rng.randint(-30, 60)}
        for key in rng.sample(["nbf", "iat", "aud", "aud"], rng.randint(1, 3)):
            if key == "aud":
                claims[key] = rng.choice([api, "web", [api], ["web", api], ["web"], []])
            else:
                claims[key] = now + rng.randint(-30, 30)
        cases.append([claims, now, api, rng.choice([0, 0, 5, 30])])
    return cases[:n]
