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
  - excess logic: `extra_condition` (a branch also requires one of its variables to be truthy);
  - function misuse: `call_swap` (a sibling call with a compatible signature: startswith/endswith,
    strip/lstrip/rstrip, find/rfind, append/extend, any/all, floor/ceil, round/int, sorted/reversed where
    the result is iterated, and similar);
  - variable misuse: `var_swap` (another variable of the same inferred type that is bound on every path);
  - value misuse: `const_sibling` (another literal of the same type from the same function).
  Bug-fix candidates are taken round-robin over operators, realistic operators first in every round, so
  operators with many sites (constants, variables) cannot crowd out the rest. `qe_mutant_kill` keeps the 12
  classic operators.
- **Real** (`REAL_SHARE` = 25%): one of the real-bug entries, ported from merged RuVector PR #1125
  (`32033a8b`). The fixed upstream code is the reference; each recorded real fault is the pre-fix logic
  from the parent commit `3e1d1886`, stored as an exact replacement in META `real_faults`. The task leads
  with a real fault and fills the remaining slots of its level with injected faults, under the same
  filters (revealing inputs, subtle share, no cancelling subsets). Real-bug entries are never drawn by the
  injected tier or by `qe_mutant_kill`. If no real fault fits a level and seed, the task falls back to the
  injected tier.

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

The oracle never runs in the same process as model-influenced code.

1. **Reference outputs** come from running the original corpus function in the sandbox. Calls are JSON
   in, JSON out.
2. **Candidate code** (a `qe_bugfix` working copy) runs in a separate child process (`python -S -s -P` with
   `PYTHONHASHSEED=0`; `-I` is not used because it ignores the hash seed, and set ordering would then differ
   between the reference and the candidate):
   - an environment set entirely by the parent, and a temporary working directory;
   - rlimits on CPU, address space, file size (0), open files and core dumps, set by the child on itself
     before any candidate code is compiled. There is no process-count limit, because RLIMIT_NPROC is
     per-user; the static gate makes forking unreachable;
   - stdout and stderr redirected away, and results returned on a private channel tagged with a nonce;
   - per-call and per-batch wall-clock timeouts.
   The child returns only the outputs for the given inputs. Expected values never enter the child.
3. **A static gate runs before execution.** It rejects imports outside an allowlist (`math`, `re`,
   `itertools`, `functools`, `collections`, `string`, `bisect`, `heapq`, `statistics`, `decimal`,
   `fractions`). It also rejects dunder attribute access and the names `open`, `exec`, `eval`, `compile`,
   `__import__`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `breakpoint`, `input`, `help`
   and `memoryview`. Code runs with a restricted builtins map. This stops the obvious hack: reading the
   reference implementation from the image.
4. **Results are compared in the parent.** Values must be JSON-equal with strict types (bool is not int,
   int is not float). Floats match to `1e-9` relative or `1e-12` absolute. An exception compares by type
   name, so a function that is specified to raise `ValueError` must keep raising it. In `qe_mutant_kill`, an
   int `expected` is read as a float wherever the original returned a float, at any depth, because JSON
   writes `2.0` as `2`. Whether `{"raises": ...}` means an exception depends on what the original actually
   did.
5. **Machine load and sandbox limits never decide an outcome.** A timed-out call is re-probed alone with a
   200x margin over the corpus's 5 ms budget; only code that still times out is treated as looping, and the
   rerun is capped at 10 s so slow code cannot multiply cost. A timeout, a crash, or a `MemoryError` or
   `RecursionError` (our 256 MiB cap and stack depth, not the code's behaviour) is never a behavioural
   difference: never a fault, never a kill, never a reason to switch function. The planned move to
   CPython-on-WASI with fuel metering makes all of this deterministic by construction (identical outcome and
   fuel across hosts and load, measured).
6. **No partial credit by accident.** Every non-empty proper subset of a task's faults must still fail some
   hidden input, so faults that cancel each other out can never let a partial fix pass. A d1 fault must be
   revealed by at least 3 distinct hidden calls, so special-casing the one reported call cannot pass.
7. **Nothing shown is ever graded.** Visible tests and the d1 report come from a second input pool that
   excludes every graded input, so outputs of the fixed edge cases are never shown and cannot be learned
   as a lookup table across episodes. In mutant-kill, the original plus the diff is
   the mutant. Values echoed by `run_tests` are capped at 200 characters each.

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
- Standard library only, and only modules from the static-gate allowlist.
- Pure and deterministic: no I/O, no time, no global state.
- Total over `gen_inputs` within 5 ms per call.
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

## Calibration and submission

- `scripts/calibrate.py` drives the environment in-process with the arena-style system prompt. It calls
  `Qwen/Qwen3.8-27B` through the HF router with `enable_thinking: false` and records each episode's
  trajectory, usage and reward.
- For free local runs, `--base-url http://localhost:11434/v1 --model qwen3.8:27b-mlx` uses Ollama. Ollama
  ignores `chat_template_kwargs`, so with `--thinking off` the script also sends `reasoning_effort: "none"`
  to any non-HF endpoint. A local MLX quantization is not the trainer's serving stack, so local numbers are
  a relative signal, not the calibration of record. `--fault-kind real|injected` forces the bug-fix fault
  source for per-tier calibration.
- The submission request is rendered and checked with the same discipline as rUv's MetaHarness entry:
  exact image digest, anonymous pull, `openenv validate`, `/schema` equality, and native replay of
  every task ID.
