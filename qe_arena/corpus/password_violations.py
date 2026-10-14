"""Check a new password against an account password policy. Original."""

from string import ascii_lowercase, ascii_uppercase, digits, punctuation

META = {
    "entry": "password_violations",
    "origin": "Original.",
    "tags": ["security", "authentication", "validation"],
    "domain": "cybersecurity",
}


def password_violations(password, username, min_length, history):
    """Return the codes of the policy rules a new password breaks, in this order, each once:

    "length": fewer than min_length characters. "classes": fewer than 3 of ASCII lowercase,
    ASCII uppercase, ASCII digits and string.punctuation occur. "username": username has 3 or
    more characters and occurs in password, ignoring case. "repeat": one character 3 or more
    times in a row. "reused": password is in history. Raises ValueError if min_length < 8.
    """
    if min_length < 8:
        raise ValueError("min_length < 8")
    found = []
    if len(password) < min_length:
        found.append("length")
    classes = sum(any(c in group for c in password)
                  for group in (ascii_lowercase, ascii_uppercase, digits, punctuation))
    if classes < 3:
        found.append("classes")
    name = username.lower()
    if len(name) >= 3 and name in password.lower():
        found.append("username")
    run = 1
    for i in range(1, len(password)):
        run = run + 1 if password[i] == password[i - 1] else 1
        if run >= 3:
            found.append("repeat")
            break
    if password in history:
        found.append("reused")
    return found


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["Tr0ub4dor&3x", "alice", 8, []], ["short1A", "bob", 8, []], ["Exactly8", "bob", 8, []],
        ["exactly8!", "bob", 8, []], ["Exactly8!", "bob", 9, []], ["alllowercase", "bob", 8, []],
        ["ALLUPPER123", "bob", 8, []], ["lower123!!", "bob", 8, []], ["Sup3rAliceX!", "alice", 8, []],
        ["Sup3rALICEx!", "Alice", 8, []], ["Sup3rAlX!x!", "al", 8, []], ["Sup3rBobX!", "bob", 8, []],
        ["Paaa55word!", "carol", 8, []], ["Paa55word!", "carol", 8, []], ["aaaB1!xyzw", "x", 8, []],
        ["xyzwB1!ccc", "q", 8, []], ["xyzB1!cccc", "q", 8, []], ["aaB1!xyza", "q", 8, []], ["Winter2026!", "dave", 8, ["Winter2026!"]],
        ["Winter2026!", "dave", 8, ["winter2026!", "Winter2025!"]], ["Pass word1", "eve", 8, []],
        ["Pa55word!", "eve", 7, []], ["", "", 8, []], ["aaa", "aaa", 12, ["aaa"]],
        ["Ünïcödé123", "bob", 8, []], ["Pa55word!", "eve", 0, []],
    ]
    names = ["alice", "Bob", "root", "al", "admin"]
    pool = "aAbB1!xX9#zzz" + "root"
    while len(cases) < n:
        name = rng.choice(names)
        pw = "".join(rng.choice(pool) for _ in range(rng.randint(4, 14)))
        if rng.random() < 0.3:
            pw = pw[:3] + name.upper() + pw[3:]
        history = [pw] if rng.random() < 0.15 else ["Old1!pass"]
        cases.append([pw, name, rng.choice([8, 8, 10, 12]), history])
    return cases[:n]
