# Workflow reference

Long-form version of the SKILL.md workflow with edge cases, transitions, and recovery.

## Phase transitions

```
Resume ───► Intent ───► Discovery ───► Scope (G1) ───► Assessment
                                                           │
                                                           ▼
                                                    Requirements (G2)
                                                           │
                                                           ▼
   Diff ◄─── Execute ◄─── Tasks ◄─────────────────────────┘
```

A skipped gate is a workflow bug, not a shortcut. If you find yourself about to call `nki_generate_kernel` and there is no `.nki-agent/requirements.md`, stop and back up to Requirements.

## State files

```
.nki-agent/
├── context.json         ← phase, intent, selected kernels, model pin
├── discovery.json       ← raw output of nki_discover_kernels
├── assessment.json      ← per-kernel skill scores + stub compile result
├── requirements.md      ← user-approved budgets
├── tasks.md             ← per-kernel task tracker
└── runs/                ← one folder per kernel attempt
    └── <kernel-id>/
        ├── candidate-1.py
        ├── compile-1.log
        ├── verify-1.json
        └── profile-1.json
```

## Edge cases

### Repo has no kernels

`nki_discover_kernels` returns an empty list. Tell the user "I didn't find any kernels in this workspace. Want to try pointing me at a specific file?" — don't proceed to Scope.

### User insists on a kernel that's already NKI

The Discovery output flags `already_nki = true`. If the user picks it anyway, ask "this is already an `@nki.jit` kernel — do you want to re-optimize it or skip?" Re-optimization runs the same flow.

### `qwen3-sft-v4` not available in the customer's region

The MCP server's `nki_generate_kernel(model="qwen3-sft-v4")` returns a region-mismatch error. Surface to the user, offer to (a) switch to `opus-4-8`, (b) switch to `vllm` failover, or (c) abort.

### Verification passes but profile is slower than baseline

The kernel is correct but not a win. Mark the task `correct_but_slow` in `tasks.md`, emit the diff with a warning, and leave the merge decision to the user.

### Compile retries exhausted

After N compile failures with the same root error, do not blindly retry. Hand off: include the source, the compiler stderr, the skill_lookup matches, and a 1-paragraph analysis of why the model is stuck.

### User aborts mid-execution

Save state to `.nki-agent/context.json` so resume works. Do not delete partial runs.

## Router policy (in AgentCore)

```
def resolve(task_spec) -> ModelBackend:
    if task_spec.model_pin:
        return backends[task_spec.model_pin]
    skills = nki_skill_lookup(task_spec.operator_signature)
    if skills.top_score >= 0.75:
        return backends["qwen3-sft-v4"]
    return backends["opus-4-8"]
```

Escalation rules:

- 2 consecutive compile fails on `qwen3-sft-v4` → next attempt uses `opus-4-8`.
- 2 consecutive verify fails on `qwen3-sft-v4` → next attempt uses `opus-4-8`.
- `opus-4-8` does NOT escalate.
- A user pin (`model_pin` set) disables escalation.
