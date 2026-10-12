# openenv-qe-arena design

A training environment for [OpenEnv Arena](https://openenvarena-arena.hf.space/). The arena trains
Qwen3.8-27B with GRPO in this environment and then scores it on 40 private tasks in 8 domains.
This environment targets the software-engineering domain with quality-engineering skills:
finding and fixing a bug, and reasoning precisely about what code does.

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

### `qe_bugfix`: repair an injected bug

A corpus function gets 1, 2 or 3 injected faults (by level). The model sees the buggy module, a short spec and
visible tests. It can read files, run the visible tests, edit with an exact string replacement, and submit.

| Level | What the model gets |
|---|---|
| d1 | One fault. An input that the bug breaks, without its expected value (derive it from the docstring). Visible tests pass |
| d2 | Two faults, at least one subtle (at most 15% of hidden inputs reveal it). No failing input |
| d3 | Three faults, at least two subtle. No failing input |

Calibration v1 (thinking off, 8 episodes per cell) showed that giving the expected value of a failing test made
the task trivial (d1 and the old d3 both 8/8), so no level shows an expected value for a failing input. Calibration v2 (12 episodes per cell) still
solved single-fault d2 12/12, so d2 and d3 now carry 2 and 3 faults. The prompt never states how many there
are, so the model has to keep checking the code against the spec instead of stopping after the first fix.

Reward: **1.0 only if every hidden differential test passes** (edge cases plus seeded random inputs,
compared against the original function's outputs). Otherwise 0. Unchanged code is 0. Partial credit is
deliberately absent: a mutant already passes most inputs, so a pass fraction would pay for doing nothing.

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
     building its result row (normalizing, serializing, `str()` of an exception), so corpus calls use 24 k
     to 1.34 M fuel (median 58 k), a 75x margin. Between calls the harness gets only 20 M
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
     refuses a literal `re.DEBUG` flag.
   - **Candidate code only ever runs on its call's fuel.** The result row is built before the end marker:
     the exception's `str()` (an instance can override what it calls), normalizing and serializing the
     value, and dropping the call's objects (a generator's `finally:` runs there). The red team finished a
     2.8e8-fuel loop inside an exception's `str()` on the harness's refill before this.
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
     - a string literal may not contain `__` or a frame-attribute name (`tb_frame`, `f_back`, `f_locals`,
       `f_globals`, `gi_frame`, …, matched as whole words);
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
   - `scripts/fingerprint.py` hashes kind, value and fuel for every corpus function on 64 inputs (2,560
     calls). `tests/test_sandbox.py` pins the result, currently `301caf2f1ee2a77f` with 194,606,462 total fuel,
     identical on arm64 macOS, an amd64 Linux host and the amd64 image run with the arena flags. The commit that
     changes the pin says why.
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
META = {"entry": "function_name", "origin": "...", "tags": ["parsing"]}

def function_name(a, b):
    """Precise spec, including edge cases and what raises."""
    ...

def gen_inputs(rng, n):
    """Return exactly n argument lists. Deterministic edge cases first, then rng-driven cases."""
```

Rules:
- Standard library only, and only modules and names the static gate allows (`sandbox.SAFE_EXPORTS`).
- Pure and deterministic: no I/O, no time, no global state.
- Cheap over `gen_inputs`: within 5 ms per call natively and far below `FUEL_PER_CALL` in the sandbox (the
  most expensive corpus call uses 1.1 M of its 100 M fuel).
- Arguments and return values are JSON-serializable: int, str, bool, None, lists, and dicts with
  string keys. Avoid floats unless they are essential.
- The entry function plus helpers is 12–45 lines.
- At least 6 mutation sites, with at least 80% of single mutants killed by `gen_inputs(rng, 64)`.

## Calibration and submission

- `scripts/calibrate.py` drives the environment in-process with the arena-style system prompt. It calls
  `Qwen/Qwen3.8-27B` through the HF router with `enable_thinking: false` and records each episode's
  trajectory, usage and reward.
- The submission request is rendered and checked with the same discipline as rUv's MetaHarness entry:
  exact image digest, anonymous pull, `openenv validate`, `/schema` equality, and native replay of
  every task ID.
- The image runs as uid 10001 with a read-only root filesystem, but it needs a writable `/tmp`
  (`docker run --read-only --tmpfs /tmp`, as CI does). The sandbox itself writes nothing. openenv imports
  gradio, which needs a temporary directory at import time, so a bare `--read-only` fails at startup;
  the image from before the WASI sandbox failed the same way.
