# Codebase — top-level directory guide

What each top-level directory in this repo does and why it exists. Each entry is
tagged with where the explanation comes from:

- **[README.md]** — taken directly from the root `README.md`.
- **[other doc]** — taken from another doc in the repo (its own README, a steering
  file, `pyproject.toml`, etc.), cited by path.
- **[inferred]** — not documented anywhere; inferred from directory contents, file
  extensions, config files, and naming conventions.

Directories that are build/cache artifacts (git-ignored, regenerated on demand) are
listed at the end under "Generated / local-only" rather than described as if they
were part of the source layout.

---

## `.claude-plugin/`

**[README.md]** Claude Code plugin manifest. Lets Claude Code install this repo's
agent as a plugin (`/plugin marketplace add ...` / `/plugin install
kernel-forge-aws-transform`), wiring up the MCP server and skill for that surface.

## `.codex-plugin/`

**[README.md]** Codex plugin manifest — the Codex-surface equivalent of
`.claude-plugin/`, so the same agent installs the same way for Codex users.

## `.kiro/`

**[README.md]/[other doc]** Kiro-native config. README.md: "Kiro consumes the same
MCP server natively — no marketplace step. The repo ships `.kiro/settings/mcp.json`
... and `.kiro/steering/nki-kernel.md`". Per `.kiro/steering/nki-kernel.md` itself,
the steering file is "a thin pointer" that activates the PyTorch/Triton→NKI agent
inside Kiro and defers to `skills/nki-kernel/SKILL.md` for the actual workflow, so
all three IDE surfaces (Claude Code, Codex, Kiro) stay behavioraly identical.
**[inferred]** `.kiro/agents/` (present but empty in this checkout) is Kiro's
convention for custom per-repo agent definitions — not populated here, so this repo
relies on the default agent plus the one steering doc.

## `mcp/`

**[README.md]** The `kernelforge-nki-mcp` Python package — "one MCP server is the
durable contract" for the whole agent. Implements the seven MCP tools
(`nki_discover_kernels`, `nki_generate_kernel`, `nki_compile`, `nki_verify`,
`nki_profile`, `nki_skill_lookup`, `nki_emit_diff`) that every IDE surface calls
through. **[other doc]** Per `mcp/README.md`, it's a stateless, customer-side
process launched locally (`uvx kernelforge-nki-mcp@latest`) that SigV4-signs its
own requests to AgentCore and the Trn1 reward server using the standard AWS
credential chain — no long-lived credentials are stored in the package itself.
Ships its own `pyproject.toml`, `uv.lock`, and `tests/` — an independent `uv`
project from the repo root.

## `skills/`

**[README.md]/[inferred]** Holds `skills/nki-kernel/`, "Skill: SKILL.md +
references/ (steering)" per the README's repo-layout tree. **[inferred]** from
its contents (`SKILL.md` plus a `references/` directory with `tile-model.md`,
`common-ops.md`, `compiler-errors.md`, `workflow.md`, `layout-rules.md`): this is
the single canonical source of the NKI-conversion workflow and hard rules,
consumed identically by all three IDE surfaces so behavior never forks between
Claude Code, Codex, and Kiro — each surface's plugin/steering config just points
at this shared skill rather than re-implementing the workflow.

## `infrastructure/`

**[README.md]** "Everything that gets deployed to AWS — see
`docs/deployment-architecture.md` for the full map." Contains the four
sub-projects that provision and run on AWS:

- `infrastructure/agentcore/` **[README.md]** — "Strands Agent + model router +
  AgentCore deploy." The application code (agent entrypoint, model router) that
  is packaged and shipped *into* the Bedrock AgentCore Runtime.
- `infrastructure/agentcore_cdk/` **[README.md]** — "CDK scaffold for the
  AgentCore runtime." An `agentcore create`-scaffolded CDK app wrapping AWS's own
  `@aws/agentcore-cdk` L3 constructs, deployed via `agentcore deploy`.
- `infrastructure/reward_server/` **[README.md]** — "Trn1 Flask service (compile
  / verify / profile / baseline)." The application code that runs on the
  Trainium (`trn1.2xlarge`) EC2 instance under systemd, doing the actual
  compile/verify/profile work against real silicon.
- `infrastructure/reward_server_cdk/` **[README.md]** — "CDK stack that
  provisions the Trn1 from scratch." A hand-written CDK app (VPC, security
  groups, the EC2 instance, PrivateLink endpoints) — no API Gateway/NLB/VPC
  Link (removed; the AgentCore runtime reaches the reward server directly,
  SG-to-SG) and no dependency on `@aws/agentcore-cdk`.**[other doc]** Per `docs/deployment-architecture.md`, this grouping exists
specifically to separate "AWS-deployed" code from "developer-workstation-only"
code — everything under `infrastructure/` ends up running in the customer's AWS
account; everything else in the repo never does.

## `cicd/`

**[other doc]** Per `cicd/README.md`: "Deployment-lifecycle scripts for the ATX
NKI Agent: prepare, security-check dependencies, deploy, destroy, and clean up
local caches/environments." Contains `common.sh` (shared path/helper constants,
sourced by the rest) plus `prep.sh`, `lint.sh`, `security-check.sh`,
`update-deps.sh`, `deploy.sh`, `destroy.sh`, and `clean.sh`. `deploy.sh` and
`destroy.sh` are explicitly flagged there as high-risk/destructive and
confirmation-gated (`CICD_YES=1` to skip prompts, e.g. in CI). This is the
workstation-side automation that *drives* deployment of everything under
`infrastructure/` — it is not itself deployed anywhere.

## `scripts/`

**[README.md]** "Local dev / eval / smoke tests" in the repo-layout tree.
**[inferred]** from contents (`eval_nkibench_agentcore.py`,
`p1_4_shape_sweep.py`, `p1_5_fix_loop_demo.py`, `rerun_blog_results_opus48.sh`,
`smoke_test.sh`, `smoke_opus48.sh`): one-off/ad-hoc developer scripts for running
benchmarks and smoke tests against an already-deployed AgentCore/reward-server
endpoint — e.g. the README's own "Run the benchmark" step
(`bash scripts/rerun_blog_results_opus48.sh`) and the sweep/demo scripts that
call the reward API directly. These
are not part of `cicd/`'s deploy lifecycle and not packaged/shipped anywhere.

## `nkibench/`

**[README.md]** "NKIGen-Bench task registry (Level 1/2/3)." **[inferred]** from
file names (`level1.py`, `level2.py`, `level3.py`, `__init__.py`): the benchmark
task definitions referenced by the paper (Tang et al., ICML 2026) and by the
headline compile/verify rates quoted in `README.md` — this is data/task
definitions for evaluation, not agent runtime code.

## `results/`

**[README.md]** "Recorded on-device measurement runs." **[inferred]** from
contents (many dated/named `*.json`/`*.details.json` files, e.g.
`nkibench150_final_opus48_2026-07-04.json`, referenced directly from
`README.md`'s results table): committed, historical evidence of actual on-device
compile/verify/profile runs backing the published percentages — "nothing is
claimed that wasn't measured on hardware" per the README's framing. These are
data artifacts, not code.

## `docs/`

**[README.md]** "Architecture diagram + design docs." Contains `architecture.md`
(+ its companion `architecture.png`, referenced from the
main README's Architecture section) describing the end-to-end solution design.
**[inferred]** also contains `deployment-architecture.md` — not mentioned in the
README's repo-layout tree (added after that tree was last written), mapping
every component to workstation-vs-AWS and the exact file path that defines it;
cross-referenced from `infrastructure/`'s own description above.

## `examples/`

**[README.md]** "Sample kernels + transcripts." **[inferred]** from contents:
top-level files (`torch_softmax_reference.py`, `triton_softmax_input.py`,
`nki_softmax_output.py`) are a minimal single-kernel before/after triple; the
`cuda_rmsnorm_migration/` subdirectory (with its own `README.md`,
`demo.sh`/`run_migration.py`, `rmsnorm_cuda_input.py`, `rmsnorm_reference.py`,
`nki_rmsnorm_output.py`, `migration_report.json`, `demo_transcript.txt`) is a
full recorded end-to-end migration example — the same CUDA→NKI RMSNorm case
cited in the README's "End-to-end integration test" section as proven on a live
`trn1.2xlarge` (`max_abs_error ≈ 2.1e-4`).

## `tests/`

**[README.md]/[inferred]** Holds `test_integration_chain.py`, explicitly
documented in the README's "End-to-end integration test" section: it "walks the
whole migration chain that a real request traverses" (`atx CLI → skill → MCP
tools → AgentCore runtime → Trn1 reward server`), with a hardware-free tier
(always on) and a live on-device tier (gated by `NKI_LIVE=1`). Alongside it,
`test_atx_cli.py` pins the AWS Transform (`atx`) CLI surface: a hardware-free
tier checks the `atx/` transformation definition + `atx/mcp.json` wiring, and a
live tier (gated by `ATX_CLI=1`) shells out to the installed `atx` binary to
confirm it enumerates the seven MCP tools. This is the
repo-root/cross-package integration-test directory, distinct from the
package-local `tests/` directories under `mcp/tests/`,
`infrastructure/agentcore/tests/`, and `infrastructure/reward_server/tests/`
(all four are wired together by the root `conftest.py`/`pytest.ini`).

## `.ash/`

**[inferred]** from `.ash/.ash.yaml`: configuration for AWS Labs'
Automated Security Helper (ASH) — the container/local static-analysis tool
(Bandit, Semgrep, Checkov, detect-secrets) used for dependency and IaC
scanning. `ash_output/` (its scan output — reports, SARIF, CSV) is
regenerated per run and is not meant to be hand-edited or treated as source.

---

## Top-level files worth calling out

- **`README.md`** **[README.md]** — the project's own front door; source for
  most of the entries above.
- **`Makefile`** **[inferred]** — dispatch layer over `cicd/*.sh` (matches the
  `Makefile`/`cicd/` split convention described generically in
  `.kiro/steering/project-scripts.md`), giving `make deploy`/`make destroy`-style
  entry points referenced directly in `README.md` and `cicd/README.md`.
- **`pyproject.toml`** **[other doc]** — declares this as the "Root dev-tooling
  project for the ATX NKI Agent repo (lint/format/test/audit only — no runtime
  package)" per its own `description` field; holds `pytest`, `ruff`,
  `pip-audit`, and `flask` (test-only) as dev dependencies. Distinct from
  `mcp/pyproject.toml` and `infrastructure/agentcore/pyproject.toml`, which are
  the two projects that actually ship runtime code.
- **`conftest.py`** / **`pytest.ini`** **[inferred]** — root-level pytest
  wiring that makes a bare `pytest` from the repo root discover and correctly
  import all four test suites (`tests/`, `mcp/tests/`,
  `infrastructure/agentcore/tests/`, `infrastructure/reward_server/tests/`) in
  one run; must stay at the repo root.
- **`requirements-agentcore.txt`** **[inferred]** — passed to
  `agentcore create --requirements ...`
  in `infrastructure/agentcore/deploy.sh`; deliberately kept at the repo root
  rather than moved into `infrastructure/agentcore/` because it crosses that
  CLI's own working-directory expectations — its ultimate fate (keep vs.
  auto-generate from `pyproject.toml`) is an open item.
- **`.mcp.json`** **[README.md]** — "MCP server descriptor (uvx-launched)" in
  the repo-layout tree; the manifest that tells any MCP-aware agent surface how
  to launch `kernelforge-nki-mcp`.
- **`LICENSE`** **[README.md]** — MIT-0, as stated in the README's License
  section.

## Generated / local-only (not part of the source layout)

**[inferred]**, from `.gitignore` and file timestamps/extensions — these exist
in a local checkout but are build artifacts, caches, or environment state, not
committed source: `.venv/`, `mcp/.venv/` (per-project virtualenvs — see
`.kiro/steering/python-coding.md`-style `uv` conventions), `__pycache__/`,
`.pytest_cache/`, `.coverage`, `chat.json` (this session's own transcript),
`.env` (local environment overrides), `SBOM.json` and `NOTICE` (regenerated by
`cicd/generate_sbom.py`), and `uv.lock`/`mcp/uv.lock` (committed, but generated —
not hand-edited).
