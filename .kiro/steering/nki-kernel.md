---
inclusion: manual
---

# NKI Kernel Optimization (Kiro steering)

This steering document activates the PyTorch/Triton → NKI kernel agent inside Kiro.
It is a thin pointer: the authoritative workflow, hard rules, and references live in
the shared Skill so all surfaces (Claude Code, Codex, Kiro) behave identically.

**Trigger this steering when the user says:** "convert this kernel to NKI",
"optimize for Trainium", "PyTorch to NKI", "Triton to NKI", "rewrite this kernel
for Inferentia", or "make this kernel run on Neuron". Do **not** use it for general
PyTorch optimization, whole-program `torch-neuronx` compilation, or non-Neuron
hardware (CUDA, ROCm).

## How to run it

1. Read the canonical workflow and NKI hard rules in [`#[[file:../../skills/nki-kernel/SKILL.md]]`](../../skills/nki-kernel/SKILL.md),
   plus the references under `skills/nki-kernel/references/`.
2. Drive the loop with the `kernelforge-nki-mcp` MCP server (configured in
   `.kiro/settings/mcp.json`): `nki_discover_kernels → nki_skill_lookup →
   nki_generate_kernel → nki_compile → nki_verify → nki_profile → nki_emit_diff`.
3. Honor the two human gates — **Scope (Gate 1)** and **Requirements (Gate 2)** —
   exactly as the Skill specifies. Never emit a diff without a passing on-device
   verification report.

Every candidate is compiled, numerically verified, and profiled on a real Trainium
device through the reward server; nothing is reported that wasn't measured on silicon.
