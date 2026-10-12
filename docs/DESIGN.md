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
     `FUEL_PER_CALL` = 100 M per call and `MODULE_FUEL` = 200 M for the module body. Corpus calls use
     8.8 k to 1.11 M fuel (median 19 k), so the per-call budget is a 90x margin. A 60 s wall-clock epoch
     deadline exists only as an infrastructure backstop.
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
   | `crash` | unserializable result, oversized output, module-level error, missing entry | no |
   | `infra` | the worker died or the wall-clock backstop fired; `timeout` is its legacy alias | no |
   | `rejected` | the static gate refused the source | no |

   An unsettled result, or an `exc` of `MemoryError` or `RecursionError` (our limits, not the code's
   behaviour), is never a behavioural difference: never a fault, never a kill, never a reason to switch
   function, and never a pass. Outcomes and fuel are deterministic, so `tasks._outputs` runs a batch once.
   The one exception is `infra`, which is the only transient kind: such a batch is run once more.
   `run_calls` also replaces a dead worker and retries once on its own. This retry keeps the generator stable
   if a worker dies mid-batch (`tests/test_review_findings.py` simulates exactly that), and it can never turn
   `fuel` or `resource` into anything else.
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
     It packs them into a zip with fixed timestamps. It refuses any output that does not match the pinned
     digests (`python.wasm` d24bd98d…, `python314.zip` 959c807f…); the Docker builder stage runs it.
   - `scripts/fingerprint.py` hashes kind, value and fuel for every corpus function on 64 inputs (2,560
     calls). `tests/test_sandbox.py` pins the result: `eb43ccdc9b4a0149` with 77,667,303 total fuel,
     identical on arm64 macOS, an amd64 Linux host and the amd64 image run with the arena flags.
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
