# openenv-qe-arena design

A training environment for [OpenEnv Arena](https://openenvarena-arena.hf.space/). The arena trains
Qwen3.8-27B with GRPO in this environment and then scores it on 40 private tasks in 8 domains.
This environment trains quality-engineering skills: finding and fixing a bug, and reasoning precisely about
what code does. Those skills do not depend on the domain, so since v4 the same mutation-tested, deterministic
graders also run on code from other arena domains (see "Domain cells").

## Constraints that shape everything

- **Arena sandbox:** 2 vCPU, 16 GiB, no GPU, and it has **outbound internet**, so keeping candidate code offline
  is our job (the static gate below). The image must be under 2 GiB compressed with one port and a `/health`
  endpoint. OpenEnv is pinned to `86a180ede21e044f7929b9a7783ad83aa67d83a3`. Rewards are in `[0, 1]`.
- **The trainer builds its system prompt from our action schema.** Field descriptions are the only text we
  control there, so they state the protocol. Each optimizer step plays 5 episodes of one task in fresh sandboxes.
  No seed is guaranteed at reset, so an unseeded reset draws a fresh instance from OS randomness. A fixed
  instance per task ID would let the model memorise it (one entrant's run saturated to zero reward variance that way).
- **An invalid reply ends the episode** and is scored through `finish_action` (`{"op": "submit", "answer": {}}`).
  For `qe_bugfix` that grades the edits made so far, so real work still counts. The unchanged file scores 0.
- **The trainer appears to run with thinking off** (inferred by rUv's team from completion lengths). With
  thinking off, Qwen3.8 echoes the action schema unless the reset prompt states the reply format with a
  concrete example. Every reset prompt here does.
- **Budgets are per episode** and count every observation after reset, so observations stay small:
  functions are 12–45 lines and test output is truncated.
- **GRPO needs mixed outcomes.** A cell the model always solves or never solves gives no gradient. We
  calibrate each family × difficulty to 1–3 successes out of 4 in thinking-off mode.
- **No free reward.** Submitting the unchanged buggy code, an empty answer or a malformed action earns 0.

## Task families

Native task IDs follow `<family>-d<1|2|3>`, for example `qe_bugfix-d2`.

### Domain cells

Every corpus entry names its arena domain in `META["domain"]`, spelled as the leaderboard spells it
(`qe_arena/domains.py`): `software-engineering`, `industrial-physical-systems`, `natural-science`,
`office-white-collar`, `finance-economics`, `mathematics-or-formal-reasoning`, `cybersecurity` or
`media-content-production`. A task ID may name a domain between the family and the level:

| Task ID | Corpus it draws |
|---|---|
| `qe_mutant_kill-d2` | software-engineering entries: exactly the pool every v1-v3 ID drew, so those IDs, tasks and exported rows are byte-identical |
| `qe_mutant_kill-finance-d2` | only `finance-economics` entries |

Slugs: `industrial`, `science`, `office`, `finance`, `math`, `security`, `media`. Software engineering has none.
`TASK_IDS` stays the ten software-engineering cells in their frozen order; `DOMAIN_TASK_IDS` follows, ordered
by domain, family and level, and `ALL_TASK_IDS` is both. A domain's cells (every family and level) are served
only once it has at least `MIN_DOMAIN_ENTRIES` (4) injected entries: with fewer, one function dominates a cell,
and a function that cannot serve a seed (no subtle mutant, no fault inside the d4 budget, too few short rows
for a flaky suite) leaves too few others to fall back on. `tests/test_domains.py` checks the rule both ways.
Since v4 every domain is served: 4 or 5 original entries each, 70 domain cells beside the 10 software ones.
Real-bug entries are software engineering, so a domain cell's `qe_bugfix` faults are always injected. A
calibration `pool` must name entries of the cell's domain. A domain cell's generator makes up to `DOMAIN_PASSES` (16) passes over its
pool before giving up, because a 4-entry pool sometimes has no function that serves a seed on one pass (for
`qe_flaky` most often: a clean shared-state suite needs two inputs with equal results). Later passes redraw
renaming, inputs and site order; over seeds 5,000,000-5,000,299, 4 of 6,300 flaky domain builds failed at 4
passes and none at 16. Software cells keep the original single pass, so their generation is unchanged, and the
real-fault tier keeps one pass because its fallback to injected faults continues on the same random stream.

The dataset export keeps the software-engineering rows in `tasks.jsonl` (byte-identical) and writes the domain
rows to `domain_tasks.jsonl` (the card's `domains` config). A row's `domain` is the arena domain with spaces
for hyphens, which is the `software engineering` the earlier rows already carry.

### `qe_bugfix`: repair an injected or real bug

A corpus function gets 1, 2 or 3 faults (by level). The model sees the buggy module, a short spec and
visible tests. It can read files, run the visible tests, edit with an exact string replacement, and submit.

Faults come from two sources, drawn per task. Nothing in the prompt or files says which:

- **Injected** (the default, 1 − `REAL_SHARE`): mutation sites on one of the 40 injected-tier functions. The
  mutation engine has 20 operators. 12 are classic one-token flips (`cmp_boundary`, `cmp_negate`,
  `arith_swap`, `const_off_by_one`, `bool_swap`, `not_drop`, `slice_shift`, `range_bound`, `return_flip`,
  `augassign_swap`, `min_max_swap`, `break_continue_swap`). 8 are realistic defect shapes from the
  HumanEvalFix/DebugBench taxonomy and RuVector PR #1125:
  - missing logic: `guard_delete` (an early-exit guard is gone), `arm_drop` (one if/elif arm of a dispatch
    is gone), `member_drop` (an allowlist literal lost a member) and `condition_drop` (one operand of an
    and/or is gone);
  - excess logic: `extra_condition` (a branch also requires one of the function's own parameters or locals
    to be truthy; never a comprehension or lambda variable);
  - function misuse: `call_swap` swaps in a sibling call with a compatible signature. Method siblings
    apply only where the receiver's type is inferred: on strings startswith/endswith, strip/lstrip/rstrip,
    find/rfind and the like; on lists append/extend, with append only when its argument is iterable.
    Other siblings are any/all, math.floor/ceil, round/int, and sorted/reversed where the result is
    iterated. Swaps that would mostly kill by raising are left out (values/keys, remove/discard), and so
    is anything on a module receiver other than math (re.split has no re.rsplit);
  - variable misuse: `var_swap` (another variable of the same inferred type that is bound on every path);
  - value misuse: `const_sibling` (another literal of the same type from the same function).
  Bug-fix candidates are taken round-robin over operators, realistic operators first in every round, so
  operators with many sites (constants, variables) cannot crowd out the rest. `qe_mutant_kill` keeps the 12
  classic operators.
- **Real** (`REAL_SHARE` = 15%, one real entry drawn uniformly per task): one of the real-bug entries, ported from merged RuVector PR #1125
  (`32033a8b`). The fixed upstream code is the reference; each recorded real fault is the pre-fix logic
  from the parent commit `3e1d1886`, stored as an exact replacement in META `real_faults`. The task leads
  with a real fault and fills the remaining slots of its level with injected faults, under the same
  filters (revealing inputs, subtle share, no cancelling subsets). Filler candidates are subsampled
  (`FILLER_KEEP`), so a small function's one subtle site cannot fill most of its real tasks. Real-bug entries are never drawn by the
  injected tier or by `qe_mutant_kill`. If no real fault fits a level and seed, the task falls back to the
  injected tier.

| Level | What the model gets |
|---|---|
| d1 | One fault. An input that the bug breaks, without its expected value (derive it from the docstring). Visible tests pass |
| d2 | Two faults, at least one subtle (at most 15% of hidden inputs reveal it). No failing input |
| d3 | Three faults, at least two subtle. No failing input |
| d4 | d3's faults plus a total edit budget (below). No failing input |

Calibration v1 (thinking off, 8 episodes per cell) showed that giving the expected value of a failing test made
the task trivial (d1 and the old d3 both 8/8), so no level shows an expected value for a failing input. Calibration v2 (12 episodes per cell) still
solved single-fault d2 12/12, so d2 and d3 now carry 2 and 3 faults. The prompt never states how many there
are, so the model has to keep checking the code against the spec instead of stopping after the first fix.

Reward: **1.0 only if every hidden differential test passes** (edge cases plus seeded random inputs,
compared against the original function's outputs). Otherwise 0. Unchanged code is 0. Partial credit is
deliberately absent: a mutant already passes most inputs, so a pass fraction would pay for doing nothing.

#### `qe_bugfix-d4`: the edit budget and the localised-repair gate

d4 is d3's fault generation plus two limits that rule out rewriting the function from its docstring. Its
random stream is `qe_bugfix-d4:<seed>`, and the task ID was appended to `TASK_IDS`, so d1-d3 and every
existing pin are byte-identical. Code: `qe_arena/edit_budget.py`.

- **Budget:** 25% of the buggy file's length, rounded up to 25 characters, counted as `len(old) + len(new)`
  over all edits. It is priced on the character-level minimal fix: changes closer than 16 characters merge
  into one edit, and each edit widens until its `old` is unique. That fix's median share of the file is
  about 0.10 (p90 0.21).
- **Localised-repair gate:** checked on every edit and again at grading. The candidate must keep:
  - the buggy file's top-level statements, in the same order (no shadow definition, no new top-level code,
    no `globals()` trick);
  - every function definition, with the same arguments and decorators (no nested shadow function);
  - no lambda beyond those already in the file;
  - every statement stays reachable. A static check proves exits through return, raise, if/else,
    try (when its body or finally ends), with, and loops that must run their body. No branch or loop may
    test something whose value cannot depend on the input: a literal or display (`{0}`, `()`, `[x]`, an
    f-string with text), anything with no variable in it (builtins and module dunders such as
    `__name__` are not variables), a name bound only by def, class or import, a walrus by its value,
    `x OP x`, `x or <truthy>`, `x in ()`, `x is []`, `isinstance(_, object)`, `len(...) >= 0`, a call
    to a builtin whose result is truthy for every argument (`type(x)`, `repr(x)`, `id(x)`,
    `ValueError(x)`) or always None (`print(x)`), a special method (`x.__class__`), and a generator
    expression. So no wrapper (`if True:`, `while 1:`, `for _ in [0]:`,
    try/finally) can bypass the old body. Tautologies are open-ended; the token allowance bounds what a
    tautology the check misses can hide;
  - a total change of at most `min(50, 0.35 × code tokens)` preorder AST tokens. The code tokens are
    counted on the shown file without docstrings.
  A compact re-implementation, even behind an always-true guard, needs more tokens than that.
- **What the model sees:** the prompt states the budget, the token allowance, the rules and the refusal
  cap. Each applied edit reports the budget left. A refused edit (over budget, or not a localised repair)
  is not applied and costs no budget. The first `MAX_FREE_REFUSALS` (4) refusals use no step; later ones do.
- **Serving:** a fault is added only if the resulting task still fits the budget and the gate. Generation
  tries other faults in the same function before moving to another function.
- **No fault leak:** the budget and the token allowance are functions of the shown file only. A budget
  derived from the faults (the calibrated 1.5x line-diff prototype) would tell the model how large the
  faulty code is.
- **Adversarial tests** (`tests/test_d4_attacks.py`, 300 seeds, attacker given the correct function):
  shadow append, early return, guarded early return, nested-def shadow, `if False:` wrap, try/finally,
  `for _ in [0]:` and `while 1:` wrappers, an inserted one-line return, lambda reassignment, decorator
  and `globals()` redefinition. Every one breaks the gate and grades 0. The oracle
  (the minimal fix) earns 1.0 on all 300 within budget, gate and the 12-step limit.

### `qe_mutant_kill`: write a test that kills a mutant

The model sees the original function and a mutant as a unified diff. It submits a test case as data:
`{"args": [...], "expected": <value>}`. No model-written code ever runs.

| Level | What the model gets |
|---|---|
| d1 | One mutant that random inputs kill often |
| d2 | One mutant that random inputs rarely kill (the input must be chosen deliberately) |
| d3 | Three subtle mutants (each revealed by at most 15% of random inputs). Up to three test cases; credit is the mean over mutants |

Reward per mutant (best over the submitted tests): 0.5 if `original(args) != mutant(args)` (the input
distinguishes them), and 1.0 if it also has `expected == original(args)` exactly (the model computed the
right answer by hand). A correct `expected` on an input that does not kill earns nothing, so easy inputs
give no free credit. A kill with a wrong `expected` still earns half, which keeps a learning gradient for
partly right reasoning. The episode reward is the mean over mutants.

### `qe_flaky`: find, diagnose and repair a flaky test

`module.py` holds a corpus function, a suite of 4-5 tests and a fixed harness `run_suite(order, seed, t0)`. The
sandbox has no clock and no `random`, so flakiness is simulated deterministically: each run builds a fresh `env`
whose seeded streams (`draw`, `frac`), fake clock (`now`) and unordered-collection order (`pool`) derive from the
run's (order, seed, t0), and module-level dicts persist between tests within a run. One test carries one injected cause
(`shared_state`, `cache_leak`, `shared_random`, `clock_boundary`, `unordered_iteration`, `float_equality`); about 15%
of tasks are clean controls whose right answer is `{"test": null}`. About a quarter of `draw`, `now` and `frac` flakes
fail only 2-10% of runs; those tasks are graded on 96 runs instead of 24.

**No surface feature names the flaky test.** Every check comes from a template with a faulty and a safe
parametrization of the same text shape (a ternary over `env.draw` whose branches return equal values, a clock bucket
compensated by the 250 ms tick, `w * 8 / 8 == w` next to `w * 3 / 3 == w`, a polluting dict write of a value with the
same result, a coarse memo key whose colliding inputs give the same answer), and decoys mostly reuse the flaky test's
own template. The audit's syntax-only classifier fell from 0.847 to 0.22-0.25 and a learned logistic regression over AST features from 0.60-0.68 to 0.52-0.55 (the always-null baseline is 0.15).

| Level | What the model does | Reward |
|---|---|---|
| d1 | Name the flaky test (or null) | 1.0 for the right test |
| d2 | Name the test and its cause | 0.5 for the test, 1.0 with the cause |
| d3 | Repair the test by edits, then name it | 0.5 for the test; 1.0 only for a repair that passes every check below |

Ground truth comes from execution, not the generator: exactly one test must both pass and fail over the graded runs and
every other test must always pass, and the cause is the single facility whose per-test reset (a grader-only harness
variant) makes that test constant.

A d3 repair is graded on the **original** tests with only the flaky test's function spliced in from the edit, so
emptying the polluter or editing the harness changes nothing. The grader runs its own copy of the harness, defined
before the tests and holding every builtin it uses as a default argument, with a fresh `Env` class per run, and it
rebuilds the function under test and the tests inside `build()` on every run, one sandbox process per graded run, so
nothing (a mutable default, a list mutated in place) survives from one run to the next; dict values are deep-copied
on every reset. The spliced tests must pass `flaky.gate_region`: no harness names, no rebinding of builtins or of the
function under test, no attribute assignment, no default arguments or decorators, and test functions take exactly
`(env)`. Full credit needs all of: every test passes on every graded run; the repaired test alone passes an
adversarial facility sweep whose runs look like graded ones (seed and t0 drawn from the task's secret stream; each
of the first eight draws forced in turn to every residue 0-7 among large random values; clocks just below powers of
ten and calendar periods; a grid of values `env.frac` can produce, walked from every start; every pool order), which
catches repairs such as `// 10**9` or `env.draw(10**9)` that keep the defect; and it fails on every graded run
against a reference mutant (classic operators only) that changes the values, never just the order, of the faulty
check's results. Shown runs (all passing) use seeds disjoint from the graded ones. Code: `qe_arena/flaky.py`
(harness, gate, grading), `flaky_gen.py` (task assembly), `flaky_templates.py` (check templates); tests:
`tests/test_flaky.py`, `tests/test_flaky_audit.py`.

## Action protocol (one schema for both families)

```json
{"op": "read", "path": "*"}
{"op": "run_tests"}
{"op": "edit", "path": "module.py", "old": "if n < lo:", "new": "if n <= lo:"}
{"op": "submit", "answer": {}}
{"op": "submit", "answer": {"tests": [{"args": [3, 7], "expected": 10}]}}
```

- `read`: one virtual file, or `*` for all files in one observation.
- `run_tests`: `qe_bugfix` only. Runs the visible tests on the current working copy. At most 3 calls.
- `edit`: `qe_bugfix` only. `old` must occur exactly once in the file. Edits accumulate.
- `submit`: ends the episode. `qe_bugfix` grades the working copy; `qe_mutant_kill` grades `answer.tests`.

At most 12 steps per episode.

### Terminal interface (`Dockerfile.terminal`, `qe_arena/terminal_env.py`)

A second image serves the `qe_bugfix` d1–d3 and `qe_mutant_kill` cells of every domain behind the action shape of
OpenEnv's `tbench2_env`, which the arena's evaluation probably uses:

```json
{"action_type": "exec", "command": "sed -n 1,40p module.py"}
{"action_type": "write_file", "file_path": "module.py", "content": "<the whole file>"}
{"action_type": "evaluate"}
```

- **Exploring is real; grading is not.** `exec` runs real bash in the episode's workspace (`qe_arena/shell.py`).
  `evaluate` reads one file, `module.py` or `answer.json`, and grades it with the same WASM graders as the JSON
  interface. Nothing else the shell does reaches the reward, and the hidden tests never exist on disk.
- **All or nothing**, like `evaluate`: a fix passes every hidden test; an answer kills every mutant with exact
  expected values. At 32 actions the episode is graded as it stands (the arena's `finish_action` is `evaluate`).
- **Isolation is the operating system's.** The server runs as root only to give each episode its own uid
  (20000–20062) and a 0700 workspace; commands, writes and the graded-file read run as that uid, under ulimits,
  with a scrubbed environment and umask 077, and every process of the uid is killed after each command (kill(-1) sent as that uid, so a fork cannot slip past a /proc scan and no recycled pid is signalled as root). `/app`
  and the WASI runtime are 0700 root, so the corpus (the reference implementations) is unreadable. The shared
  world-writable directories are locked, so the workspace is the only writable place. An unprivileged seccomp
  filter allows only AF_UNIX sockets (no network, not even loopback to the server) and denies io_uring,
  System V IPC and POSIX message queues. A uid whose cleanup cannot be verified is never reused. Tested as
  root inside the image on native x86_64 and arm64 (`tests/test_shell_isolation.py`, CI job `terminal-image`).
  Under amd64 emulation on an Apple-silicon Mac, seccomp is unavailable and the shell selects disabled mode.
- **Fail closed.** If setuid is unavailable (a runtime that drops CAP_SETUID, board #213), the start-up log says
  `shell isolation mode: disabled`, `exec` is refused, and `write_file` and `evaluate` still work on validated
  relative paths. Calibration (`calibrate.py --interface terminal`) refuses to run anywhere but uid mode, so a
  model's commands never run on a developer machine.
- Not isolated, for want of privileges (no cgroups or namespaces): total disk use and inode count of a
  workspace (only per file), total memory across an episode's processes (only per process), the argv of
  programs a command starts (`/proc/<pid>/cmdline`; the command text itself travels in the private
  environment), and abstract AF_UNIX sockets. The last two let concurrently running episodes cooperate if both
  try; none of them reaches the grader, the hidden tests or the reference sources.

## Verifier integrity

The oracle never runs in the same process as model-influenced code, and no code we grade ever runs natively.

1. **Reference outputs** come from running the original corpus function in the same sandbox as the
   candidate. Calls are JSON in, JSON out. Never compare a native result with a sandbox result: the guest is
   CPython 3.14 on wasm32 (4-byte pointers, 32-bit `hash()`), the server is CPython 3.12 on 64-bit.
2. **Candidate code runs in CPython-on-WASI under wasmtime fuel.** `qe_arena/sandbox.py` sends each batch over
   a pipe to a long-lived worker process (`qe_arena/wasi_worker.py`), so the server never loads wasmtime and a
   wasmtime fault or host bug cannot take down `/health`. The worker runs `python.wasm` (brettcannon/cpython-
   wasi-build v3.14.7) in wasmtime 47.0.1, with a fresh Store and Instance per batch:
   - **Fuel, not time.** Every call is metered in wasmtime fuel (about one unit per wasm instruction):
     `FUEL_PER_CALL` = 100 M per call and `MODULE_FUEL` = 200 M for the module body. A call's fuel includes
     building its result row (normalizing, serializing, `str()` of an exception) and releasing what the call
     left behind (a full `gc.collect()` of objects created since setup, which the harness freezes), so corpus
     calls use 542 k to 2.05 M fuel (median 580 k), a 49x margin. Between calls the harness gets only 20 M
     (`HARNESS_FUEL`); no candidate code runs there. A 60 s wall-clock epoch deadline exists only as an
     infrastructure backstop.
   - **Memory and stack.** Linear memory is capped at 256 MiB (`DEFAULT_MEM_MB`); the wasm stack is bounded.
   - **No host.** The guest has no host directory at all: `qe_arena/wasi_fs.py` answers every `fd_*` and
     `path_*` call from memory, exposing one read-only directory `/lib` that holds one reproducible stdlib zip,
     with canonical metadata (fixed times, fixed inode numbers, nlink 1). wasmtime's own preopen passed host
     inode numbers, link counts, directory sizes and real access times into the guest; the access time was
     effectively the host clock, and because the interpreter stats its stdlib at startup the values shifted
     heap layout and fuel between a host and a container on the same machine. The environment holds only
     `PYTHONHASHSEED=0`. Clocks and clock resolution are fixed or stepped, `random_get` is a seeded SHAKE-256
     stream, `poll_oneoff` is unsupported, and there are no sockets, so the guest cannot reach the network even
     though the arena machine can.
   - **Call boundaries.** The guest harness (`qe_arena/guest_harness.py`) writes `B`, `E` and `R` markers to
     fd 1, each tagged with a per-batch random nonce. The host sets the call's fuel at `B`, reads what was
     used at `E` and takes the result row from `R`. Order is enforced, and anything else on fd 1 fails the
     batch closed, so nothing a candidate prints can forge a result or move fuel accounting.
   - **Restricted guest.** Inside, code runs with a restricted builtins map, and `import` returns only proxy
     modules that carry an audited list of names. Expected values never enter the worker or the guest.
     Python-level `sys.stdout`/`sys.stderr` discard everything (only the harness writes fd 1, with
     `os.write`), so `re.DEBUG`'s parse-tree printout or a warning cannot break the protocol. The gate also
     refuses a literal `re.DEBUG` flag, and the `re` proxy clears that bit from any flags value.
   - **Why the nonce is not the barrier.** Harness and candidate share one interpreter, and stock
     `python.wasm` has no host import that Python code can call. So anything the harness can say to the
     host, a candidate holding a writer could say too, nonce included: a red-team run with the gate
     deliberately bypassed forged a protocol-valid stream. The barrier is that no writer exists: `os`/`sys`
     are unreachable, there is no string-driven attribute lookup, stdout is discarded, and `re` cannot print.
     What a forged stream could add is also nothing beyond `return`: a value the candidate chose, or the name
     of an exception it could raise. The nonce and the strict marker order only make accidental output fail
     closed.
   - **Candidate code only ever runs on its call's fuel.** The result row is built before the end marker:
     the exception's `str()` (an instance can override what it calls), normalizing and serializing the
     value, dropping the call's objects and collecting cycles (a generator's `finally:` runs there). The
     resource flag is checked only after that cleanup, so a `RecursionError` raised and caught in a `finally:`
     during release still makes the call `resource`. The red team finished a 2.8e8-fuel loop inside an
     exception's `str()` on the harness's refill before this.
   - **Nothing worth reading above the candidate.** Candidate code runs below exactly two frames: the closure
     the harness's setup returns, and the root module frame, whose globals are cleared first. The nonce
     exists only inside `itertools.cycle` objects (no attribute or repr shows their bytes), the pending
     inputs only inside a list iterator, and the request, `sys.argv` and `sys.orig_argv` are gone. A test
     with the gate switched off walks every ancestor frame and `sys` and finds neither the nonce nor a
     pending input.
   - **Resource failures cannot be caught away.** Before any candidate code runs, the harness registers a
     `sys.monitoring` RAISE callback under its own tool id. The candidate cannot reach it: `sys` is not
     importable and no traversal route exists. It flags every `MemoryError` or `RecursionError` raised
     during a call or the module body, including ones raised inside C calls and ones the candidate catches
     with `except Exception`. A flagged call is `resource` whatever it returned, so the memory cap can
     never choose a settled value. Cost: about 3,100 fuel per raised exception, +1.8% over the corpus.
   - **One rule after a trap.** A batch runs in one instance, in order. A trap (fuel, wasm stack, any other
     trap) ends the instance. The call it interrupted gets `fuel`, `resource` or `crash`, and every later call
     is `skipped`; nothing re-runs on reconstructed state. A `MemoryError` the guest survives is not a trap,
     so later calls still run on real state. Isolating every call in its own instance would cost about 290 M
     fuel of interpreter startup per call, so instead `grade_mutant_kill` runs each submitted test as its
     own batch: credit does not depend on test order, and a test that loops one mutant cannot cancel
     another test.
   - **Every wait has a deadline, and the pool is fair.** Acquiring a pool worker, starting one, writing the
     request (non-blocking, select-bounded) and reading the response all end by `BATCH_BACKSTOP_S`. A
     stopped or wedged worker is killed and replaced, and the batch is retried once. `warm()` returns every
     worker it started even when a later one fails. Each environment instance grades under its own session
     key. Waiters are served first come first served among sessions under their cap, and no session holds
     more than `pool_size - 1` workers. Together with one instance per trapped batch, two sessions grading
     looping code kept honest episodes at a median of 3.1 s and a max of 8.6 s on ruos (2 workers), where
     the red team had measured 45 s and 154 s.
   - **Runtime identity.** At startup the worker hashes `python.wasm` and the stdlib zip, compares them with
     `qe_arena/runtime_pins.json`, and compiles and serves those exact bytes. It refuses to report ready on a
     mismatch.
   - **Workers.** One worker per CPU the container may use: the cgroup v2 `cpu.max` quota (then cgroup v1,
     the affinity mask, `os.cpu_count()`), clamped to 1..4. `QE_ARENA_WASI_WORKERS` overrides it. The app
     starts and warms them in a background thread; each compiles `python.wasm` once (about 5 s each at
     `--cpus=2`, 10.4 s for both, measured in the image), while `/health` keeps answering.
3. **A static gate runs before execution.** It rejects imports outside an allowlist (`math`, `re`,
   `itertools`, `functools`, `collections`, `string`, `bisect`, `heapq`, `statistics`, `decimal`,
   `fractions`, `typing`, `operator`) and any `from X import Y` whose `Y` is not exported. It rejects dunder
   names and attributes, attributes that reach frames, code or tracebacks (`gi_`, `cr_`, `ag_`, `f_`, `tb_`,
   `co_`, any leading `_`), class definitions, `global`/`nonlocal`, `match`, async code and oversize or overly
   deep source. It bans the names `open`, `exec`, `eval`, `compile`, `__import__`, `globals`, `locals`,
   `vars`, `getattr`, `setattr`, `delattr`, `breakpoint`, `input`, `help`, `memoryview`, `exit`, `quit` and
   `MemoryError` (so a guest `MemoryError` can only come from the memory cap).
   - **No string-driven attribute lookups.** `str.format` resolves `{0.attr}` and `{0[key]}` at runtime, out of
     reach of the AST checks. The Codex review used it to read the nonce
     (`'{0.__traceback__.tb_frame.f_back.f_locals[nonce]}'.format(e)`), the batch inputs and real module
     globals (`'{0.__class__.__init__.__globals__[sys]}'.format(typing.List)`). So:
     - a string that reaches a format path (the literal receiver of `.format`, the literal spec of
       `format()`) may not contain `__` or a frame-attribute name (`tb_frame`, `f_back`, `f_locals`,
       `f_globals`, `gi_frame`, …, matched as whole identifier words). Other literals (docstrings,
       `x.split('__')`, dict keys) are free: none of them can drive an attribute lookup, because `getattr`,
       `vars`, `globals`, `format_map`, `string.Formatter` and `operator.attrgetter` are unavailable;
     - `.format` needs a string-literal receiver whose fields are plain positions or names, with no `.` or
       `[` even in nested specs;
     - `format()` needs a literal spec and cannot be aliased.

     Ordinary formatting (`"{:.2f}"`, `"{0}-{1}"`, named fields, `"{0:{1}}"`, f-strings, `%`) still works.
   - `SAFE_EXPORTS` is frozen as literals for the 3.14 guest. Every export was audited for anything that reads
     or writes an attribute or evaluates code from a runtime string, or that hands back a module. So
     `functools.update_wrapper`/`wraps`/`singledispatch`, `string.Formatter`, `operator.attrgetter`/
     `methodcaller`/`call`, and `typing.get_type_hints`/`get_args`/`ForwardRef`/`evaluate_forward_ref`/
     `TypeVar`/`NamedTuple`/`Protocol` are absent, as are `typing`'s re-exported modules (`sys`,
     `functools`, `collections`, `types`, `copyreg`, `operator`). The gate also denies those helper names as
     attributes.
   - `tests/test_sandbox.py` walks every public attribute reachable from the exports inside the guest (depth
     4, about 2,400 objects) and asserts that no module, frame, code object or loader is among them. Exporting
     `typing.sys` on purpose makes that walk report it.
   - The walk deliberately does not follow dunders: every export's dunder graph reaches a module
     (`math.floor.__self__` is `math`, and any Python function's `__globals__` is its module), so no export
     list could pass such a walk. The defence is that no route to a dunder exists: the AST gate blocks them as
     attributes, and the formatting rules above block the only string-driven lookup.
4. **Results are compared in the parent.** Values must be JSON-equal with strict types (bool is not int,
   int is not float). Floats match to `1e-9` relative or `1e-12` absolute. An exception compares by type
   name, so a function that is specified to raise `ValueError` must keep raising it. In `qe_mutant_kill`, an
   int `expected` is read as a float wherever the original returned a float, at any depth, because JSON
   writes `2.0` as `2`. Whether `{"raises": ...}` means an exception depends on what the original actually
   did. Fuel is never compared: a call's fuel depends on what ran earlier in the same instance.
5. **Sandbox limits never decide an outcome, and machine load cannot.** Each call ends as one of these kinds:

   | Kind | Meaning | Settled? |
   |---|---|---|
   | `ok` | returned a JSON value | yes |
   | `exc` | raised; compared by type name (`RecursionError` stays unsettled, as below) | yes |
   | `fuel` | used its whole fuel budget (loops, or far too slow) | no |
   | `resource` | hit the memory cap or exhausted the wasm stack | no |
   | `crash` | unserializable result, oversized output, module-level error, missing entry, protocol violation; a candidate failure, never infra | no |
   | `skipped` | not run: an earlier call in the batch trapped and ended the instance | no |
   | `infra` | the worker died, hung or failed its identity check, or the backstop fired; `timeout` is its legacy alias | never graded |
   | `rejected` | the static gate refused the source | no |

   `resource` covers `MemoryError` and `RecursionError` however the code handled them (see the RAISE monitor
   above). An unsettled result is never a behavioural difference: never a fault, never a kill, never a
   reason to switch function, and never a pass. Outcomes and fuel are deterministic, so `tasks._outputs`
   runs a batch once.

   `infra` is the one transient kind, and it is never scored. `run_calls` replaces a failed worker and
   retries once, and `tasks._outputs` reruns an `infra` batch once more. If it is still `infra`, `_outputs`
   raises `sandbox.InfraError`. The environment lets that propagate without consuming the step or the test
   run and without ending the episode. OpenEnv's WebSocket server then answers with an `error` frame
   (`EXECUTION_ERROR`) and no reward, and the client can resend the same action. A persistent sandbox
   failure therefore never becomes a 0 that GRPO would learn from. The retries keep task generation stable
   when a worker dies mid-batch, and they can never turn `fuel` or `resource` into anything else.
6. **No partial credit by accident.** Every non-empty proper subset of a task's faults must still fail some
   hidden input, so faults that cancel each other out can never let a partial fix pass. A d1 fault must be
   revealed by at least 3 distinct hidden calls, so special-casing the one reported call cannot pass.
7. **Nothing shown is ever graded.** Visible tests and the d1 report come from a second input pool that
   excludes every graded input, so outputs of the fixed edge cases are never shown and cannot be learned
   as a lookup table across episodes. In mutant-kill, the original plus the diff is
   the mutant. Values echoed by `run_tests` are capped at 200 characters each.
8. **Every host computes the same thing, and the test suite pins it.**
   - `scripts/build_wasi_runtime.py` checks the release's sha256 and compiles the stdlib inside `python.wasm`
     to unchecked-hash pycs, in sorted path order (compileall follows readdir order, which changed one pyc).
     It packs them into a zip with fixed timestamps. It refuses any output that does not match
     `qe_arena/runtime_pins.json` (`python.wasm` d24bd98d…, `python314.zip` 959c807f…), which the worker also
     checks at startup. The Docker builder stage runs it. Both base images are pinned by digest, and wasmtime
     installs with `--require-hashes` (`requirements-wasmtime.lock`).
   - `scripts/fingerprint.py` hashes kind, value and fuel for every corpus function of a domain on 64 inputs.
     `tests/test_sandbox.py` pins the software-engineering result, currently `f50dbb1dd3917498` with
     1,732,616,699 total fuel over 45 functions (2,880 calls), identical on arm64 macOS, an amd64 Linux host and
     the amd64 image run with the arena flags, and one pin per other domain (`DOMAIN_FINGERPRINTS`), so a
     domain's new entries move only its own line. The commit that changes a pin says why.
   - Regenerate the pin only when the runtime, the guest harness, the budgets, `SAFE_EXPORTS` or the corpus
     change, and say why in the commit.
9. **Signed task manifests.** Because grading is deterministic, `qe_arena/manifest.py` describes a task by
   hashes only. It records:
   - sha256 of `python.wasm`, the stdlib zip, the guest harness and the corpus file, plus the wasmtime version;
   - the task ID and seed;
   - sha256 of the hidden inputs, the reference outputs with their fuel, the grading sources and the
     observation;
   - the fuel budgets and the memory cap.

   `scripts/sign_manifest.py` generates keys, signs and verifies offline with Ed25519 (`cryptography`).
   `verify_task` checks the signature, rebuilds the task from (task_id, seed) and compares every field, so a
   validly signed but wrong manifest still fails. The server never imports the module and never holds the
   private key; the public key belongs at `qe_arena/keys/manifest_ed25519.pub`. Signed manifests are meant to
   ship with the HF dataset export.

Items 4 to 7 came from independent review of the grader: an adversarial Codex review, a red team that built
exploits, and mutation testing of the verifier (305 mutants; the first suite caught 72.1%, the hardened
suite 97.5%). Each confirmed finding has a regression test in `tests/test_review_findings.py` or
`tests/test_reward_integrity.py`.

## Corpus contract

Each corpus entry is one file `qe_arena/corpus/<name>.py`:

```python
"""One-line summary. Adapted from <owner/repo>@<sha>:<path> (MIT)."""   # or "Original." when not adapted
META = {"entry": "function_name", "origin": "...", "tags": ["parsing"], "domain": "software-engineering"}

def function_name(a, b):
    """Precise spec, including edge cases and what raises."""
    ...

def gen_inputs(rng, n):
    """Return exactly n argument lists. Deterministic edge cases first, then rng-driven cases."""
```

Rules:
- `domain` is one of `qe_arena.domains.DOMAINS`. Code for a domain is realistic code of that domain (a loan
  schedule, a ramp limiter), not a software utility with domain names.
- Standard library only, and only modules and names the static gate allows (`sandbox.SAFE_EXPORTS`).
- Pure and deterministic: no I/O, no time, no global state.
- Cheap over `gen_inputs`: within 5 ms per call natively and far below `FUEL_PER_CALL` in the sandbox (the
  most expensive corpus call uses 1.1 M of its 100 M fuel).
- Arguments and return values are JSON-serializable: int, str, bool, None, lists, and dicts with
  string keys. Avoid floats unless they are essential.
- The entry function plus helpers is 12–45 lines.
- No string that Qwen's tokenizer reads as a control token (`<|im_end|>`, `<|endoftext|>`, `<think>`,
  `<tool_call>` and the like) in the code or the generated inputs. The policy cannot write such a string as
  text: it ends the reply, so an edit or an expected value containing one is cut off mid-JSON. Local
  calibration hit this on a real-bug entry; GGUF token names are written without the `|` delimiters.
- At least 6 mutation sites, with at least 80% of single mutants killed by `gen_inputs(rng, 64)`.

A real-bug entry also has `META["real_faults"]`: a list of `{"id", "kind", "summary", "origin", "find",
"replace"}`. `origin` is `owner/repo@<pre-fix sha>:<path>` of the code that had the bug, and `find` must
occur exactly once in the function source. The docstring is the spec and comes from the fixed code's own
docs and tests, never from the diff. Each real fault must be revealed by at least `MIN_REVEALING_D1`
distinct graded inputs, so its deterministic edge cases include enough revealing calls.

Spec sufficiency: for every real-bug entry, `tests/test_real_bugs.py` holds implementations written from
the docstring alone, one per plausible literal reading (for example "an int" with and without bools). Each
reading must equal the reference on 100 × 48 generated inputs. A served fault differs from the reference on
a graded input, so no served fault can match a literal reading of the docstring. Third-party code is
credited in `THIRD_PARTY_NOTICES`, which the image ships with the corpus. A fault whose `note` is set
reproduces the pre-fix behaviour rather than copying pre-fix code.

## Calibration and submission

- `scripts/calibrate.py` drives the environment in-process with the arena-style system prompt. It calls
  `Qwen/Qwen3.8-27B` through the HF router with `enable_thinking: false` and records each episode's
  trajectory, usage and reward.
- For free local runs, `--base-url http://localhost:11434/v1 --model qwen3.8:27b-mlx` uses Ollama. Ollama
  ignores `chat_template_kwargs`, so with `--thinking off` the script also sends `reasoning_effort: "none"`
  to any non-HF endpoint. A local MLX quantization is not the trainer's serving stack, so local numbers are
  a relative signal, not the calibration of record. `--fault-kind real|injected` forces the bug-fix fault
  source for per-tier calibration. The default task list is the software-engineering cells; `--domain-cells`
  adds every domain cell, and `--each-corpus` pins each cell to each function of its own domain.
- The submission request is rendered and checked with the same discipline as rUv's MetaHarness entry:
  exact image digest, anonymous pull, `openenv validate`, `/schema` equality, and native replay of
  every task ID.
- The image runs as uid 10001 with a read-only root filesystem, but it needs a writable `/tmp`
  (`docker run --read-only --tmpfs /tmp`, as CI does). The sandbox itself writes nothing. openenv imports
  gradio, which needs a temporary directory at import time, so a bare `--read-only` fails at startup;
  the image from before the WASI sandbox failed the same way.

## Daily flywheel (`scripts/flywheel.py`, logic in `scripts/qe_flywheel/`)

One submission per day, and one arena run is noise, so the flywheel keeps three disciplines. The
package lives under `scripts/`, not `qe_arena/`, so none of it ships in the image.

- **Which episodes count** is one rule, `stats.episode_status`, shared by `stats`, `select` and
  `onecell-search`. A `reward: null` row (harness error) and a `global_budget` row (the whole calibration
  run ran out of tokens) are excluded, never scored as 0. A `context_budget` or `completion_budget` row is
  a real failure when the episode ran at or above the submitted template's limit (the arena would fail
  it too) and is excluded, with a warning, when it ran under it. A row may record the limits it ran under
  as `episode_completion_tokens` / `episode_context_tokens`; without them, calibrate.py's documented
  defaults (4096 / 8192) are assumed, which are tighter than the template's (6144 / 12288), so pass
  `--episode-completion-tokens 6144 --episode-context-tokens 12288` and record them to measure under the
  submitted limits. `step_cap` is always a failure because the arena caps steps too.
- **`stats`** reports Jeffreys posteriors per cell and per (cell, corpus function): the measured
  same-seed informative rate, and `p_mix@4` / `p_mix@5`, the posterior-mean chance that 4 or 5 fresh
  instances do not all get the same reward (a Dirichlet over the observed reward values, so partial
  credit counts). The arena plays 5 episodes of one task per step and trains on the first 4 with a valid
  reward; its contract has no seed and an unseeded reset here draws a fresh instance, so same-seed groups
  understate mixing. A repeated `(task, seed, corpus, attempt)` key across files is a hard error.
- **`stats`** also pools each arena domain's cells (each row judged by its own cell's limits), so the
  domains the arena scores can be compared. **`domain-template`** extends a task template with the domain
  cells, each copying the limits of its software-engineering base cell (same family and level); a cell
  whose base cell is not in the template is skipped and named.
- **`select`** keeps a cell when its posterior-mean `p_mix@4` is at least `--min-mixed` (0.15), then
  Thompson-samples by the default objective `projected` (`p_mix@G`, G = 4; `--objective group` uses the
  same-seed posterior). The gate is a posterior mean, so evidence matters: 0/8 scores 0.18 and stays
  eligible as exploration, 0/16 and 16/16 score 0.10 and are excluded, 15/16 scores 0.29. Limits are
  copied verbatim from the template and the measure is recorded in `.why.json`. When no cell qualifies the
  command exits 4 and writes no task list.
- **`ledger`** is an append-only, hash-chained JSONL of arena runs. Each entry keeps the leaderboard
  record it came from and is re-validated on every read; the head is anchored in `ledger.jsonl.head`
  (written atomically), so editing or deleting any line, the last included, fails. The chain and the head
  catch accidents only: anyone who can write the ledger can rewrite it and recompute both, so publish the
  head hash `show` prints if the ledger is evidence. After a crash between an append and the head write,
  `ledger repair-head --i-checked-the-ledger` recomputes the head once a human has inspected the lines.
  `fetch --user Profa-QE [--submission-id ID]` reads the public leaderboard (8 MiB cap, every runs entry
  must be an object) and appends that user's new runs (idempotent by `run_id`, exact user match) only with
  a local receipt that is `recorded` with `post_attempted` true. It refuses a paged response: the cursor
  format could not be verified (the Space's page never sends one), unless `--cursor-url` names it. The
  leaderboard's baseline and all its runs are kept, versioned by fetch time, in `ledger-baselines.jsonl`
  and `ledger-population.jsonl`. `add` enters a record by hand as `source: manual`; `show` leaves manual
  entries out unless `--include-manual` and refuses to mix benchmarks.
- **`ledger show`** prints, in this order:
  1. each run against the UNTRAINED baseline (4/40 today), total and per domain: did training help?
  2. its rank among other entrants' runs on the same benchmark (every user's run except the ledger's
     fetched users and its own runs): the mid-rank percentile, P(a random other run >= it) and per-domain
     deltas against their task rates. This is a rank, not a test of whether training helped, since every
     entrant trained. On 2026-10-12, 38 other heldout-v2 runs averaged 2.95/40 (sd 1.14, range 1-5),
     below the baseline: training usually costs industrial and office solves. The leaderboard has one
     evaluation per submission, so evaluation noise cannot be separated from real differences between
     runs; the spread of other runs' totals is only an upper bound on it.
  3. the best-per-domain composite (selection-inflated, not a run score) against the best-per-domain of
     k DISTINCT other runs (exact over every k-subset up to 200,000 of them, else a seeded Monte Carlo);
  4. as a sensitivity, the model-based null of an untrained baseline: each domain's rate drawn from
     Beta(s + .5, n - s + .5) given the baseline's single attempt and shared by the k runs. It is generous
     (E = 6.67 for one run) because the Jeffreys prior puts about 8% on the domains the baseline scored 0
     in, so it is also shown with those domains fixed at 0, .02 and .05, with an empirical-Bayes prior
     from the other runs' pooled solves, and as a plug-in binomial lower reference.
- **`onecell-search`** proposes exactly one knob change (Darwin's one-knob idea, reimplemented). Its
  default metric is `mixed_rate`, the unbiased chance that G = 4 valid fresh instances of a cell are not
  all equal, averaged per difficulty level and weighted by the mix. A leniency knob or a mix shift toward
  easy cells raises solve rates, but on this metric it only wins where it moves a cell toward mixed
  outcomes, so `difficulty_mix` may only be scored this way. Plans are 64 seeds x 1 attempt per split for
  every task; both arms must be independent runs covering every planned `(task, seed, attempt)`, and runs
  identical apart from timings are refused. For `mixed_rate` an excluded row is dropped, as the trainer
  keeps the first 4 valid episodes of 5 (for the reward metrics it counts as 0 for that arm), and arms
  whose excluded rates differ by more than 0.02 are refused, so an arm cannot win by erroring. `decide`
  accepts only when the paired, task-stratified bootstrap lower bound exceeds the margin (0) on discovery
  and -0.05 on confirmation. Each move spends alpha = 5% across its tests: 2.5% the first time, then 1.25%,
  ... A rejected move is retired only when its discovery bootstrap upper bound is below the margin;
  otherwise it may be retested at the next, smaller alpha. With iid arms tests hold false acceptance at
  or below 5%; a three-cell gain to p = .5 is accepted at least 80% of the time. The plan's sha256 catches
  an accidental edit between propose and decide, not a deliberate one. Nothing here calls a model.

**Calibrating for the trainer's regime.** Run `scripts/calibrate.py --attempts 1 --seeds N` (N >= 16 per
cell): one attempt on each of many fresh instances is what the projection and `mixed_rate` assume. Use
`--attempts >= 2` only to study seed difficulty itself.
