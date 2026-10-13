"""Match an IPv4 address against a CIDR allowlist. Original."""

META = {
    "entry": "ip_allowlist_match",
    "origin": "Original.",
    "tags": ["security", "network", "access-control", "parsing"],
    "domain": "cybersecurity",
}


def parse_ipv4(text):
    """Return a dotted-quad IPv4 address as a 32-bit integer."""
    parts = text.split(".")
    if len(parts) != 4:
        raise ValueError("an IPv4 address has four parts")
    for part in parts:
        if not part.isdigit() or (len(part) > 1 and part[0] == "0") or int(part) > 255:
            raise ValueError("bad IPv4 octet: " + part)
    return sum(int(part) << (24 - 8 * i) for i, part in enumerate(parts))


def ip_allowlist_match(ip, allowlist):
    """Return the first allowlist entry whose network contains ip, or None.

    Addresses are dotted-quad IPv4: exactly four parts, each a decimal number 0..255 written
    without a sign and without leading zeros ("0" itself is fine, "01" is not). An entry is
    "a.b.c.d/len" with 0 <= len <= 32, or a bare address meaning "/32". Host bits set in an
    entry's address are ignored, so "10.1.2.3/8" covers all of 10.0.0.0/8. Entries are checked
    in order and the search stops at the first match, so entries after it are not validated.
    Raises ValueError if ip is malformed or a checked entry is malformed.
    """
    address = parse_ipv4(ip)
    for entry in allowlist:
        network, slash, length = entry.partition("/")
        if slash and (not length.isdigit() or int(length) > 32):
            raise ValueError("bad prefix length in " + entry)
        bits = int(length) if slash else 32
        mask = 0xFFFFFFFF ^ (0xFFFFFFFF >> bits)
        if (address & mask) == (parse_ipv4(network) & mask):
            return entry
    return None


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["10.1.2.3", ["10.0.0.0/8"]], ["11.0.0.1", ["10.0.0.0/8"]], ["192.168.1.7", ["192.168.1.7"]],
        ["192.168.1.8", ["192.168.1.7"]], ["192.168.1.8", ["192.168.1.7/31"]],
        ["192.168.1.6", ["192.168.1.7/31"]], ["8.8.8.8", ["0.0.0.0/0"]], ["8.8.8.8", ["0.0.0.0/1"]],
        ["200.1.1.1", ["0.0.0.0/1"]], ["10.9.9.9", ["10.1.2.3/8"]], ["172.16.5.4", ["172.16.0.0/12"]],
        ["172.32.0.1", ["172.16.0.0/12"]], ["1.2.3.4", ["9.9.9.9", "1.2.3.0/24", "1.2.0.0/16"]],
        ["1.2.3.4", ["1.2.3.0/24", "garbage"]], ["1.2.3.4", ["garbage", "1.2.3.0/24"]],
        ["1.2.3.4", ["1.2.3.0/33"]], ["1.2.3.4", ["1.2.3.0/32"]], ["1.2.3.4", ["1.2.3.4/"]],
        ["1.2.3.4", ["1.2.3.0/x"]], ["1.2.3.4", ["1.2.3.0/+24"]], ["1.2.3.4", ["1.2.3.0/ 24"]],
        ["192.168.1.6", ["192.168.1.7"]], ["1.2.3.4", ["1.2.3.5/32"]], ["1.2.3.4", ["9.2.3.4/32"]], ["1.2.3.4", []], ["01.2.3.4", ["0.0.0.0/0"]],
        ["0.0.0.0", ["0.0.0.0/32"]], ["255.255.255.255", ["255.255.255.254/31"]],
        ["256.1.1.1", ["0.0.0.0/0"]], ["1.2.3", ["0.0.0.0/0"]], ["1.2.3.4.5", ["0.0.0.0/0"]],
        ["1.2.-3.4", ["0.0.0.0/0"]], ["128.0.0.0", ["127.255.255.255/1"]],
    ]
    while len(cases) < n:
        ip = ".".join(str(rng.choice([0, 1, 10, 127, 128, 192, 255, rng.randint(0, 255)])) for _ in range(4))
        allow = []
        for _ in range(rng.randint(1, 4)):
            net = ".".join(str(rng.choice([0, 10, 128, 192, rng.randint(0, 255)])) for _ in range(4))
            allow.append(net + "/" + str(rng.randint(0, 32)) if rng.random() < 0.8 else ip)
        cases.append([ip, allow])
    return cases[:n]
