"""cybersecurity desk family (level 4): brute-force incidents from a synthetic sshd auth.log.

See ``desk_science`` for what a desk family is. Real auth logs carry real people's addresses and user names, so the
log is synthetic, but shaped like a Debian/Ubuntu bastion's /var/log/auth.log: year-less syslog timestamps (here
crossing a New Year), sshd's paired "Invalid user" and "Failed password for invalid user" lines, pam_unix lines that
shadow each failure, rsyslog's "message repeated N times" compression, IPv6 sources, and cron/sudo noise.

The generator plants, per request, an incident that only a sliding window finds (it straddles a clock-aligned
window boundary), one whose span is exactly the window, a slow scanner that never reaches the threshold inside a
window, a burst that only reaches it through a repeated message, an invalid-user burst, an IPv6 burst, a January
burst, and an incident whose source had logged in successfully before (but not after) the burst.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from collections.abc import Iterator

from . import Built, Family
from ._common import rows, write_csv

DOMAIN = "cybersecurity"
LEVEL = 4
FILE = "auth.log"
OUTPUT = "incidents.csv"
HEADER = ["ip", "burst_start", "failures", "succeeded"]
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
YEARS = (2022, 2023, 2024, 2025)
THRESHOLDS = (4, 5, 6, 8)
WINDOWS = (2, 5, 10, 15)
HOSTS = ("bastion-01", "edge-gw", "jump-eu1")
USERS = ("root", "deploy", "alice", "bob", "git", "backup", "ubuntu", "ansible")
INVALID = ("admin", "test", "oracle", "pi", "support", "guest", "ftpuser", "postgres", "user1", "hadoop", "nagios")
STAMP = re.compile(r"^([A-Z][a-z]{2}) {1,2}(\d{1,2}) (\d\d):(\d\d):(\d\d) \S+ (.*)$")
FAILED = re.compile(r"Failed password for (?:invalid user )?(\S*) from (\S+) port \d+")
FAILED_VALID_ONLY = re.compile(r"Failed password for (\S+) from (\S+) port \d+")
INVALID_LINE = re.compile(r"Invalid user (\S*) from (\S+) port \d+")
PAM_FAILURE = re.compile(r"pam_unix\(sshd:auth\): authentication failure;.* rhost=(\S+)")
REPEATED = re.compile(r"message repeated (\d+) times: \[ ?(.*)\]$")
ACCEPTED = re.compile(r"Accepted \S+ for (\S+) from (\S+) port \d+")
IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# ---------------------------------------------------------------- solving

def _lines(text: str, year: int, variant: str | None) -> Iterator[tuple[dt.datetime, str]]:
    last_month = None
    for line in text.splitlines():
        m = STAMP.match(line)
        if not m:
            continue
        month = MONTHS.index(m.group(1)) + 1
        if last_month is not None and month < last_month and variant != "year_fixed":
            year += 1
        last_month = month
        yield dt.datetime(year, month, int(m.group(2)), *(int(m.group(i)) for i in (3, 4, 5))), m.group(6)


def _failure(message: str, variant: str | None) -> str | None:
    """The source address if ``message`` is a failure under ``variant``'s reading, else None."""
    pattern = FAILED_VALID_ONLY if variant == "skip_invalid_user" else FAILED
    if (m := pattern.search(message)) and not (variant == "skip_invalid_user" and m.group(1) == "invalid"):
        return m.group(2)
    if variant == "count_invalid_lines" and (m := INVALID_LINE.search(message)):
        return m.group(2)
    if variant == "count_pam" and (m := PAM_FAILURE.search(message)):
        return m.group(1)
    return None


def _first_burst(times: list[dt.datetime], threshold: int, window: int, variant: str | None) -> dt.datetime | None:
    span = dt.timedelta(minutes=window)
    if variant == "total_not_window":
        return times[0] if len(times) >= threshold else None
    if variant == "fixed_window":
        buckets: dict[int, list[dt.datetime]] = {}
        for t in times:
            key = (t.toordinal() * 86400 + t.hour * 3600 + t.minute * 60 + t.second) // (window * 60)
            buckets.setdefault(key, []).append(t)
        hits = [b[0] for b in buckets.values() if len(b) >= threshold]
        return min(hits) if hits else None
    for i in range(len(times) - threshold + 1):
        gap = times[i + threshold - 1] - times[i]
        if gap < span or (gap == span and variant != "window_exclusive"):
            return times[i]
    return None


def _authlog_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    failures: dict[str, list[dt.datetime]] = {}
    successes: dict[str, list[dt.datetime]] = {}
    for when, message in _lines(files[FILE], request["year"], variant):
        count = 1
        if m := REPEATED.search(message):
            count, message = (1 if variant == "repeat_once" else int(m.group(1))), m.group(2)
        if (ip := _failure(message, variant)) is not None:
            if variant == "ipv4_only" and not IPV4.match(ip):
                continue
            failures.setdefault(ip, []).extend([when] * count)
        elif m := ACCEPTED.search(message):
            successes.setdefault(m.group(2), []).append(when)
    out = []
    for ip, times in failures.items():
        times.sort()
        start = _first_burst(times, request["threshold"], request["window_minutes"], variant)
        if start is None:
            continue
        after = [t for t in successes.get(ip, []) if variant == "success_anytime" or t >= start]
        out.append([ip, start.isoformat(), len(times), "yes" if after else "no"])
    out.sort(key=lambda r: (r[1], r[0]))
    return write_csv(HEADER, out)


def _authlog_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].strip() != ",".join(HEADER):
        return [False]
    out = rows(text)
    if len(out) != len(reference):
        return [False]
    results = []
    for got, ref in zip(out, reference):
        got = {k: (v or "").strip() for k, v in got.items() if k is not None}
        results += [got["ip"] == ref["ip"], got["burst_start"] == ref["burst_start"],
                    got["failures"] == ref["failures"], got["succeeded"] == ref["succeeded"]]
    return results or [True]


# ---------------------------------------------------------------- generation

class _Log:
    def __init__(self, rng: random.Random, host: str):
        self.rng, self.host = rng, host
        self.entries: list[tuple[dt.datetime, int, str]] = []
        self.used: set[str] = set()

    def say(self, when: dt.datetime, text: str) -> None:
        self.entries.append((when, len(self.entries), text))

    def pid(self) -> int:
        return self.rng.randint(1200, 65000)

    def port(self) -> int:
        return self.rng.randint(1024, 65535)

    def ip(self, v6: bool = False) -> str:
        rng = self.rng
        while True:
            if v6:
                ip = f"2001:db8:{rng.randrange(1, 0xffff):x}::{rng.randrange(1, 0xffff):x}"
            else:
                net = rng.choice(("203.0.113", "198.51.100", "192.0.2", f"45.{rng.randint(10, 250)}.{rng.randint(0, 255)}",
                                  f"185.{rng.randint(10, 250)}.{rng.randint(0, 255)}"))
                ip = f"{net}.{rng.randint(2, 254)}"
            if ip not in self.used:
                self.used.add(ip)
                return ip

    def fail(self, when: dt.datetime, ip: str, *, invalid: bool | None = None, repeat: int = 0) -> int:
        """One failed attempt (plus ``repeat`` compressed copies); returns the failures it stands for."""
        rng = self.rng
        invalid = rng.random() < 0.4 if invalid is None else invalid
        pid, port = self.pid(), self.port()
        user = rng.choice(INVALID if invalid else USERS)
        pre = when - dt.timedelta(seconds=rng.randint(0, 2))
        if invalid:
            self.say(pre, f"sshd[{pid}]: Invalid user {user} from {ip} port {port}")
            if rng.random() < 0.7:
                self.say(pre, f"sshd[{pid}]: pam_unix(sshd:auth): check pass; user unknown")
                self.say(pre, f"sshd[{pid}]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 "
                              f"tty=ssh ruser= rhost={ip}")
            text = f"Failed password for invalid user {user} from {ip} port {port} ssh2"
        else:
            if rng.random() < 0.7:
                self.say(pre, f"sshd[{pid}]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 "
                              f"tty=ssh ruser= rhost={ip}  user={user}")
            text = f"Failed password for {user} from {ip} port {port} ssh2"
        self.say(when, f"sshd[{pid}]: {text}")
        if repeat:
            self.say(when + dt.timedelta(seconds=rng.randint(0, 1)), f"sshd[{pid}]: message repeated {repeat} times: [ {text}]")
            # the compressed copies are logged at the repeat line's time, never after the next attempt
        tail = when + dt.timedelta(seconds=1 if repeat else 0)
        if rng.random() < 0.5:
            closing = (f"Connection closed by invalid user {user} {ip} port {port} [preauth]" if invalid else
                       f"Connection closed by authenticating user {user} {ip} port {port} [preauth]")
            self.say(tail, f"sshd[{pid}]: {closing}")
        elif rng.random() < 0.3:
            self.say(tail, f"sshd[{pid}]: Received disconnect from {ip} port {port}:11: Bye Bye [preauth]")
        return 1 + repeat

    def accept(self, when: dt.datetime, ip: str, user: str | None = None) -> None:
        rng = self.rng
        pid, port, user = self.pid(), self.port(), user or rng.choice(USERS[1:])
        method = rng.choice(("publickey", "publickey", "password"))
        key = f": ED25519 SHA256:{rng.getrandbits(128):032x}" if method == "publickey" else ""
        self.say(when, f"sshd[{pid}]: Accepted {method} for {user} from {ip} port {port} ssh2{key}")
        self.say(when, f"sshd[{pid}]: pam_unix(sshd:session): session opened for user {user}(uid={rng.randint(1000, 1010)}) "
                       "by (uid=0)")
        end = when + dt.timedelta(minutes=rng.randint(1, 90))
        self.say(end, f"sshd[{pid}]: Received disconnect from {ip} port {port}:11: disconnected by user")
        self.say(end, f"sshd[{pid}]: Disconnected from user {user} {ip} port {port}")
        self.say(end, f"sshd[{pid}]: pam_unix(sshd:session): session closed for user {user}")

    def text(self) -> str:
        out = []
        for when, _, message in sorted(self.entries):
            out.append(f"{MONTHS[when.month - 1]} {when.day:>2} {when:%H:%M:%S} {self.host} {message}")
        return "\n".join(out) + "\n"


def _spread(rng: random.Random, start: dt.datetime, seconds: int, n: int) -> list[dt.datetime]:
    """``n`` sorted times from ``start`` over exactly ``seconds`` (first at start, last at start + seconds)."""
    inner = sorted(rng.randint(1, seconds - 1) for _ in range(max(0, n - 2)))
    return [start] + [start + dt.timedelta(seconds=s) for s in inner] + [start + dt.timedelta(seconds=seconds)]


def _authlog_build(level: int, rng: random.Random) -> Built:
    year = rng.choice(YEARS)
    threshold, window = rng.choice(THRESHOLDS), rng.choice(WINDOWS)
    span = window * 60
    log = _Log(rng, rng.choice(HOSTS))
    begin = dt.datetime(year, 12, rng.randint(29, 30), rng.randint(0, 12), rng.randint(0, 59), rng.randint(0, 59))
    end = dt.datetime(year + 1, 1, rng.randint(1, 2), rng.randint(12, 23), 0, 0)
    total = int((end - begin).total_seconds())
    newyear = dt.datetime(year + 1, 1, 1)

    def at(lo: dt.datetime | None = None, hi: dt.datetime | None = None) -> dt.datetime:
        lo, hi = lo or begin + dt.timedelta(hours=1), hi or end - dt.timedelta(hours=1)
        return lo + dt.timedelta(seconds=rng.randrange(int((hi - lo).total_seconds())))

    def burst(start: dt.datetime, ip: str, n: int, seconds: int, **kw) -> None:
        for t in _spread(rng, start, seconds, n):
            log.fail(t, ip, **kw)

    # Straddles a clock-aligned window boundary: no aligned bucket holds threshold failures.
    edge = at().replace(second=0, microsecond=0)
    edge -= dt.timedelta(minutes=edge.minute % window)
    half, step, ip = threshold // 2, max(1, (span - 2) // threshold), log.ip()
    for t in [edge - dt.timedelta(seconds=1 + j * step) for j in range(half)] + \
             [edge + dt.timedelta(seconds=j * step) for j in range(threshold - half)]:
        log.fail(t, ip)
    # Exactly the window wide; nothing else from that source.
    burst(at(), log.ip(), threshold, span)
    # A slow scanner: many failures, never threshold of them inside one window.
    ip, t = log.ip(), at(hi=begin + dt.timedelta(hours=8))
    for _ in range(threshold + rng.randint(2, 6)):
        log.fail(t, ip)
        t += dt.timedelta(seconds=span // (threshold - 1) + rng.randint(30, 600))
    # Reaches the threshold only through a "message repeated" line.
    ip, t = log.ip(), at()
    log.fail(t, ip, invalid=False, repeat=threshold - 1)
    # An invalid-user burst, an IPv6 burst, and bursts across and after midnight on New Year's Eve.
    burst(at(), log.ip(), threshold + rng.randint(0, 3), rng.randint(threshold, span - 1), invalid=True)
    burst(at(), log.ip(v6=True), threshold + rng.randint(0, 4), rng.randint(threshold, span - 1))
    burst(newyear - dt.timedelta(seconds=rng.randint(span // 3, span // 2)), log.ip(), threshold, span - 5)
    burst(at(newyear + dt.timedelta(hours=1)), log.ip(), threshold + rng.randint(0, 6), rng.randint(threshold, span - 1))
    # Logged in before its burst, never after; and one that gets in after its burst.
    ip, t = log.ip(), at(lo=begin + dt.timedelta(hours=10))
    log.accept(t - dt.timedelta(hours=rng.randint(2, 8)), ip)
    burst(t, ip, threshold + rng.randint(0, 3), rng.randint(threshold, span - 1))
    ip, t = log.ip(), at(hi=end - dt.timedelta(hours=6))
    burst(t, ip, threshold + rng.randint(0, 5), rng.randint(threshold, span - 1))
    log.accept(t + dt.timedelta(seconds=span + rng.randint(10, 3600)), ip, rng.choice(("root", "ubuntu", "admin")))

    # Background: more brute force, stray failures below the threshold, legitimate logins, cron and sudo.
    for _ in range(rng.randint(2, 5)):
        burst(at(), log.ip(rng.random() < 0.3), threshold + rng.randint(0, 12), rng.randint(threshold, span * 3),
              repeat=0)
    office = [f"10.20.{rng.randint(0, 9)}.{rng.randint(2, 254)}" for _ in range(6)]
    for _ in range(rng.randint(25, 45)):
        ip = log.ip(rng.random() < 0.2)
        t = at()
        for _ in range(rng.randint(1, threshold - 1)):
            log.fail(t, ip)
            t += dt.timedelta(seconds=rng.randint(5, 4000))
        if rng.random() < 0.15:
            log.accept(t + dt.timedelta(seconds=rng.randint(5, 600)), ip)
    for _ in range(rng.randint(30, 50)):
        ip = rng.choice(office)
        if rng.random() < 0.2:
            log.fail(at(), ip, invalid=False)
        log.accept(at(), ip)
    t = begin.replace(minute=17, second=1)
    while t < end:
        pid = log.pid()
        log.say(t, f"CRON[{pid}]: pam_unix(cron:session): session opened for user root(uid=0) by (uid=0)")
        log.say(t + dt.timedelta(seconds=1), f"CRON[{pid}]: pam_unix(cron:session): session closed for user root")
        t += dt.timedelta(hours=1)
    for _ in range(rng.randint(4, 10)):
        who, t = rng.choice(("alice", "bob")), at()
        if rng.random() < 0.3:
            log.say(t, f"sudo: pam_unix(sudo:auth): authentication failure; logname={who} uid=1000 euid=0 "
                       f"tty=/dev/pts/0 ruser={who} rhost=  user={who}")
        log.say(t + dt.timedelta(seconds=4), f"sudo:    {who} : TTY=pts/0 ; PWD=/home/{who} ; USER=root ; "
                                             "COMMAND=/usr/bin/systemctl restart nginx")
    log.entries = [e for e in log.entries if begin <= e[0] <= end]

    request = {"year": year, "threshold": threshold, "window_minutes": window}
    files = {FILE: log.text(), "request.json": json.dumps(request, indent=2) + "\n"}
    statement = "\n".join([
        f"The SOC wants brute-force incidents from {FILE}, a syslog export from an SSH bastion. request.json gives "
        "year (the year of the first line), threshold and window_minutes.",
        "",
        "Rules:",
        "- Timestamps carry no year (Mmm dd HH:MM:SS, the day padded with a space). Lines are in time order. The "
        "first line is in year, and the year goes up by one whenever the month goes back (December to January).",
        "- A failure is a line \"Failed password for USER from IP port P ssh2\", including the form \"Failed "
        "password for invalid user USER from IP ...\". The \"Invalid user USER from IP\" line and the pam_unix "
        "\"authentication failure\" lines that go with a failed attempt are not further failures.",
        "- \"message repeated N times: [ MESSAGE]\" means MESSAGE occurred N more times, at that line's time: a "
        "repeated failure adds N failures.",
        "- Sources are IPv4 or IPv6 addresses; write them exactly as logged.",
        "- A source is an incident when threshold of its failures fall within window_minutes: some failure and the "
        "next threshold-1 failures of that source are at most window_minutes apart (last minus first, both ends "
        "included). The window slides; it is not aligned to the clock.",
        "- burst_start is the time of the first failure of the earliest such run, as YYYY-MM-DDTHH:MM:SS. failures "
        "is the source's total number of failures in the whole log. succeeded is yes when an \"Accepted ... for USER "
        "from IP\" line for that source is at or after burst_start, otherwise no.",
        "",
        "Write incidents.csv with exactly this header line: " + ",".join(HEADER) + ", then one row per incident, "
        "sorted by burst_start, then ip. The log is too long to print: use a script.",
    ])
    built = Built(files=files, statement=statement, output=OUTPUT, reference=None)
    built.reference = rows(_authlog_solve(files, level, None))
    return built


FAMILIES = (
    Family("authlog", DOMAIN, _authlog_build, _authlog_check, _authlog_solve,
           {trap: (LEVEL,) for trap in ("repeat_once", "count_invalid_lines", "skip_invalid_user", "count_pam",
                                        "fixed_window", "window_exclusive", "total_not_window", "year_fixed",
                                        "ipv4_only", "success_anytime")},
           levels=(LEVEL,)),
)
