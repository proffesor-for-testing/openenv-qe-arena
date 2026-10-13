"""Join an untrusted relative path under a base directory without traversal. Original."""

META = {
    "entry": "safe_join",
    "origin": "Original.",
    "tags": ["security", "path-traversal", "paths"],
    "domain": "cybersecurity",
}


def safe_join(base, user_path):
    """Resolve an untrusted relative path under the directory base without escaping it.

    root is base without trailing "/" characters. user_path is split on "/" and on "\\";
    empty and "." segments are skipped and ".." drops the last kept segment. Returns
    root + "/" + the kept segments joined by "/", or root ("/" when root is "") if none is kept.
    Raises ValueError if base does not start with "/", user_path starts with "/" or "\\",
    user_path contains a NUL character, or a ".." has no kept segment to drop.
    """
    if not base.startswith("/"):
        raise ValueError("bad base")
    if user_path.startswith("/") or user_path.startswith("\\") or "\x00" in user_path:
        raise ValueError("bad path")
    kept = []
    for segment in user_path.replace("\\", "/").split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not kept:
                raise ValueError("escapes base")
            kept.pop()
        else:
            kept.append(segment)
    root = base.rstrip("/")
    if not kept:
        return root or "/"
    return root + "/" + "/".join(kept)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["/srv/files", "report.pdf"], ["/srv/files/", "a/b/c.txt"], ["/", "etc/passwd"],
        ["/srv/files", "../etc/passwd"], ["/srv/files", "a/../../b"], ["/srv/files", "a/b/../c"],
        ["/srv/files", "a/./b//c/"], ["/srv/files", ""], ["/srv/files", "."], ["/", ""],
        ["/", "a/.."], ["/srv/files", "/etc/passwd"], ["/srv/files", "\\windows\\win.ini"],
        ["/srv/files", "a\\..\\..\\b"], ["/srv/files", "a\\b"], ["/srv/files", "ok\x00.png"],
        ["srv/files", "a"], ["", "a"], ["/srv//", "x/y"], ["/srv/files", "..."],
        ["/srv/files", "..a/b.."], ["/srv/files", "a/b/c/../../.."], ["/srv/files", "a/b/c/../../../.."],
        ["/srv/files", "./"],
    ]
    parts = ["a", "b", "img", "..", ".", "", "x.txt", "\\", "..\\.."]
    while len(cases) < n:
        path = "/".join(rng.choice(parts) for _ in range(rng.randint(1, 6)))
        cases.append([rng.choice(["/srv/files", "/srv/files/", "/", "/data"]), path])
    return cases[:n]
