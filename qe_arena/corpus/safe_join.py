"""Join an untrusted relative path under a base directory without traversal. Original."""

META = {
    "entry": "safe_join",
    "origin": "Original.",
    "tags": ["security", "path-traversal", "paths"],
    "domain": "cybersecurity",
}


def safe_join(base, user_path):
    """Resolve an untrusted relative path under the directory base, refusing to escape it.

    base is an absolute POSIX directory such as "/srv/files"; trailing "/" characters on it are
    ignored, so "/srv/files/" and "/srv/files" are the same root and "/" is the filesystem root.
    user_path is split on both "/" and "\\" (a backslash is a separator too); empty segments and
    "." are skipped, and ".." removes the most recently kept segment. Returns the root, then "/",
    then the kept segments joined by "/". When no segment is kept it returns the root itself
    ("/" for the filesystem root).
    Raises ValueError if base does not start with "/", if user_path starts with "/" or "\\"
    (an absolute path), if it contains a NUL character, or if a ".." has no kept segment to remove.
    """
    if not base.startswith("/"):
        raise ValueError("base must be absolute")
    if user_path.startswith("/") or user_path.startswith("\\") or "\x00" in user_path:
        raise ValueError("absolute path or NUL byte in user path")
    kept = []
    for segment in user_path.replace("\\", "/").split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not kept:
                raise ValueError("path escapes the base directory")
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
