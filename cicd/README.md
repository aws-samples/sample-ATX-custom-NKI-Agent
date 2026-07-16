# cicd/

Deployment-lifecycle scripts for the ATX NKI Agent: prepare, security-check
dependencies, deploy, destroy, and clean up local caches/environments. Every
script sources `common.sh` and can be run standalone from anywhere — paths
are resolved relative to the repo root, not the caller's working directory.

## Scripts

| Script | Purpose |
|---|---|
| `common.sh` | Shared helpers (`section`, `have`, `confirm`) and path constants. Sourced by every other script; not run directly. |
| `npm_audit_filtered.sh` | Runs `npm audit --json` for a given CDK app dir and filters out the fixed allowlist of known, no-fix-available advisories documented in "Known npm advisory exclusions" below, via `npm_audit_filter.py`. Used by `security-check.sh`/`update-deps.sh` for `agentcore_cdk/`; always advisory (exits 0), never changes the pass/fail gate. |
| `prep.sh` | Install/sync all dependencies (`uv sync` for the repo **root** dev-tooling project plus `mcp/` and `infrastructure/agentcore/`, `npm install` + `npm run build` for the CDK apps) and run the full pytest suite via `uv run pytest --cov`. |
| `lint.sh` | Lint every Python project (`ruff check`, default settings) — root, `mcp/`, `infrastructure/agentcore/`. |
| `security-check.sh` | Dependency vulnerability scanning: `pip-audit` per Python project — root dev-tooling, `mcp/`, `infrastructure/agentcore/` (**fatal** — drives the exit code), `npm audit` per CDK app (filtered for `agentcore_cdk/`, see "Known npm advisory exclusions" below) and an ASH pass over the whole repo (both **advisory**). |
| `update-deps.sh` | Update dependencies across all three `uv` projects (`uv lock --upgrade` + `uv sync`), `infrastructure/reward_server/requirements.txt` (floor-version bump via `update_requirements_txt.py`, preserving its `>=` pin style and comments), and both CDK apps (`npm update`), then re-run the same fatal/advisory vulnerability scan as `security-check.sh` (plus a `pip-audit -r` pass over the updated `requirements.txt`), and regenerate `SBOM.json` + `NOTICE` via `generate_sbom.py`. |
| `deploy.sh` | Deploy AgentCore first, then the reward server (`infrastructure/reward_server_cdk`), then wire them together via `repoint-agentcore.sh`. **High-risk** — creates live, billed AWS resources (a `trn1.2xlarge`, VPC). Confirmation-gated. Before touching any CDK command, validates `agentcore_cdk/atxnkiagent/agentcore/agentcore.json` + `aws-targets.json` if they already exist (JSON syntax, a non-empty target with a 12-digit account id, and an `nki_agent` runtime entry) — see "Existing but broken `agentcore.json`/`aws-targets.json`" below. |
| `destroy.sh` | Tear down everything `deploy.sh` created, in reverse order. **Destructive.** Confirmation-gated. |
| `clean.sh` | Remove local caches and build artifacts. Non-destructive by default (keeps `.venv`/`node_modules`); `--deep` also removes those. |

## Usage

```bash
./cicd/prep.sh              # install deps, run tests
./cicd/lint.sh               # ruff check across all Python projects
./cicd/security-check.sh    # scan dependencies for known vulnerabilities
./cicd/update-deps.sh       # upgrade deps (uv + npm), then re-scan
./cicd/deploy.sh            # deploy (prompts before each stage)
./cicd/destroy.sh           # tear down (prompts before each stage)
./cicd/clean.sh             # clear caches only
./cicd/clean.sh --deep      # also remove .venv/node_modules (re-run prep.sh after)
```

Set `CICD_YES=1` to skip all confirmation prompts (for CI / non-interactive
use) — this matches the `agentcore` CLI's own `--yes` convention:

```bash
CICD_YES=1 ./cicd/deploy.sh
```

`deploy.sh` takes no input beyond AWS credentials. The reward server is
reachable only via a direct security-group rule scoped to the AgentCore
runtime's ENIs (SG-to-SG) — there is no API Gateway, no endpoint-type toggle,
and no IAM grant to manage for that path.

```bash
CICD_YES=1 ./cicd/deploy.sh                   # unattended
```

## Root dev-tooling project

The repo root has a `pyproject.toml` (`package = false` — no runtime code,
no wheel built) that pins the shared dev tools used by these scripts:
`pytest`, `pytest-cov`, `ruff`, `pip-audit`, and `flask` (test-only, for the
reward-server auth/validation branches). `uv sync` at root creates a `.venv`
for these; `prep.sh` and `security-check.sh` both run through it via
`uv run`.

This is **separate** from `mcp/` and `infrastructure/agentcore/`'s own `pyproject.toml` +
`uv.lock` — those pin each project's actual *runtime* dependencies (`mcp`,
`boto3`, `bedrock-agentcore`, `strands-agents`, etc.) in their own venvs and
stay independent by design. `infrastructure/reward_server/` remains `requirements.txt`-only
(installed via CodeArtifact on the Trn1 itself, not from a dev machine) and
is intentionally not part of the root `uv` project.

## Deploy ordering: AgentCore first, single reward_server_cdk pass

`deploy.sh` deploys **AgentCore first**, then wires it to the reward server,
in a single `cdk deploy` pass per stack:

1. **Deploy AgentCore.** This repo has **two** AgentCore deployment
   surfaces with no single declared canonical path — `infrastructure/agentcore/deploy.sh`
   (the `bedrock-agentcore-cli` pip package, resolved via `uv run agentcore`
   from `infrastructure/agentcore/pyproject.toml`'s dev group, against `infrastructure/agentcore/app.py`)
   and `infrastructure/agentcore_cdk/atxnkiagent/` (a full `agentcore create`-scaffolded
   project, using the `@aws/agentcore` **npm** package — a different tool
   with the same command name — pinned as a local devDependency in
   `agentcore/cdk/package.json` and resolved by its `node_modules/.bin`
   path, not a global install). `deploy.sh` drives the latter when present,
   because that's the path `repoint-agentcore.sh` (step 4) already patches
   automatically; it falls back to `infrastructure/agentcore/deploy.sh` otherwise, but that
   path needs `REWARD_SERVER_URL` exported manually after step 3, since
   `repoint-agentcore.sh` won't wire it up. This inconsistency predates
   these scripts and isn't something they resolve — picking one path is
   an open recommendation.
2. **Resolve the execution role ARN** via `aws bedrock-agentcore-control
   list-agent-runtimes` (by name) then `get-agent-runtime` (for `roleArn`)
   — the AWS API directly, not by parsing `agentcore status` text/JSON
   output, which doesn't expose the execution role ARN as a documented
   field.
3. **Single `reward_server_cdk` deploy** (Trn1, SG-to-SG, no gateway — there
   is no IAM grant to attach). The `-c agentCoreExecutionRoleArn=<arn>`
   context from step 2 is passed through for compatibility with callers
   that expect it; the stack itself does not act on it — access is gated
   entirely by security groups.
4. **`repoint-agentcore.sh`** patches AgentCore's `REWARD_SERVER_URL` and
   VPC network config to point at the now-provisioned reward server, and
   redeploys the runtime in place.

**What AgentCore-first ordering buys**: access between the two stacks is
gated entirely by security groups, with no IAM grant to sequence around.
Deploying AgentCore first matters because `repoint-agentcore.sh` (step 4)
patches AgentCore's runtime *in place*, which requires the runtime to
already exist. AgentCore's own `REWARD_SERVER_URL`/VPC config isn't known
until *after* the reward server exists, so AgentCore is deployed once with
a placeholder, then redeployed once, in place, at step 4, once the real
values are known. This ordering puts that redeploy on the side that's fast
and code-only (AgentCore), instead of `reward_server_cdk` (a much heavier
stack: VPC, PrivateLink, and a `trn1.2xlarge`) needing two full passes.

## Existing but broken `agentcore.json`/`aws-targets.json`

Both files live under `infrastructure/agentcore_cdk/atxnkiagent/agentcore/`,
are gitignored (they hold a real AWS account id once generated), and are
normally regenerated automatically by `deploy.sh` on a fresh checkout. If a
prior run was interrupted or cancelled partway through, they can be left on
disk but incomplete — most commonly `aws-targets.json` as an empty `[]`, or
`agentcore.json`'s `runtimes` array missing the `nki_agent` entry. Because
every `cdk` CLI invocation (via `cdk.json`'s `"app": "node dist/bin/cdk.js"`)
loads this project's own app entrypoint — `cdk bootstrap` included, not just
`deploy`/`synth` — a broken `aws-targets.json` used to surface only once CDK
itself failed, opaquely, with:

```
AgentCore CDK synthesis failed: No deployment targets configured. Please define targets in agentcore/aws-targets.json
```

`deploy.sh` now validates both files (JSON syntax, a non-empty
`aws-targets.json` entry with a valid 12-digit `account`, and an `nki_agent`
entry in `agentcore.json`'s `runtimes`) **before** calling into any `cdk`
command. If validation fails on an existing pair of files, it prints the
specific reason(s) and prompts:

- **`r` / regenerate** (recommended, and what `CICD_YES=1` picks
  automatically for unattended/CI runs) — deletes both files and lets the
  existing from-scratch scaffold step re-create them via a temporary
  `agentcore create`, patched with the real account id/region and the
  `nki_agent` runtime name. Safe: neither file is deployed state, only local
  config.
- **`e` / edit** — leaves the files untouched and exits, so you can fix them
  by hand. See
  `infrastructure/agentcore_cdk/atxnkiagent/agentcore/.llm-context/aws-targets.ts`
  for `aws-targets.json`'s exact schema (`name`/`description`/`account`/`region`).
- **`a` / abort** — exits without changing anything.

A from-scratch checkout (neither file present yet) is unaffected by this
check — that case was already handled by the existing scaffold step.

## Security-check policy

- **`pip-audit --local`** (per Python project) is **fatal** — a known CVE in
  a shipped dependency fails the script (and, in CI, the pipeline).
- **`npm audit`** and **ASH** are **advisory** — reported for visibility,
  never block. **ASH runs in `--mode local` unconditionally.** Local mode
  uses a different, weaker-isolation scanner set and
  needs no Docker/Podman on the machine running this script. It's a
  deliberate tradeoff for this fast pre-deploy gate — a more thorough
  container-mode ASH pass (Docker, falling back to Podman) is recommended
  for a full periodic review, run separately from this script.
- **`npm audit` for `infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/`
  runs through `npm_audit_filtered.sh`, not raw `npm audit`** — see "Known
  npm advisory exclusions" below for why, and for the exact list.
  `infrastructure/reward_server_cdk/` still runs raw `npm audit` directly
  (currently 0 findings; nothing to filter).

## Known npm advisory exclusions

`infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/`'s `npm audit`
consistently reports 16 advisories (as of `@aws/agentcore@0.24.1`) across
`esbuild`, `hono`, `qs`, `@babel/core`, `@opentelemetry/core`,
`brace-expansion`, `fast-xml-parser`, and `js-yaml`. All 16 share the same
root cause and the same "no local fix" conclusion, so
`npm_audit_filtered.sh` excludes them from the "needs attention" output
(they are still printed, under an "Excluded known advisory hit(s)" heading —
nothing is silently hidden) rather than have them re-triaged as new findings
on every `security-check.sh`/`update-deps.sh` run.

**Why these can't be fixed locally.** Every one of these packages is nested
inside `@aws/agentcore`'s own published npm tarball
(`node_modules/@aws/agentcore/node_modules/...`), not declared in this
repo's `package.json` and not reachable by `npm update`/`npm audit fix` —
confirmed via `npm ls <pkg>`, which shows each sitting under
`@aws/agentcore`'s own dependency tree (its bundled CLI tooling: the agent
inspector's `hono`/`express`/`qs`-backed local dev server, its bundled
OpenTelemetry exporters, its bundled AWS SDK clients, and its own bundled
lint/format toolchain — `eslint-plugin-react-hooks`'s `@babel/core`,
`@textlint`/`@trivago/prettier-plugin-sort-imports`'s `js-yaml`/
`brace-expansion`). `@aws/agentcore@0.24.1` is the latest version published
to npm at the time of writing — there is no newer release to bump to.
`npm audit fix --force`'s own resolver, when asked, reports the only path
that changes any of these is downgrading `@aws/agentcore` to `0.10.0`
(`isSemVerMajor: true` in its audit JSON) — three months and 14 releases
behind current, which would regress the CLI's own features/fixes, not
remediate a vulnerability. This is not something `security-check.sh`/
`update-deps.sh` can or should apply automatically.

**Real-world exploitability.** `@aws/agentcore` is a local developer CLI
(`bin: agentcore` in its own `package.json`) — this repo doesn't deploy it as
a running AWS service. The vulnerable packages back its local `agentcore
dev`/inspector tooling: tracing the bundled CLI (`dist/cli/index.mjs`) shows
its proxy calls target `hostname: "127.0.0.1"` and its own port-probe helper
checks `127.0.0.1` before falling back to `0.0.0.0`. Realistic exposure is
therefore limited to a developer's own machine or shared dev network during
an active `agentcore dev`/inspector session — not internet-facing, and not
reachable through anything `infrastructure/agentcore_cdk`,
`infrastructure/reward_server_cdk`, or `infrastructure/agentcore` actually
provisions in AWS.

**Reported upstream.** These are being reported to the `aws/agentcore-cli`
maintainers (`https://github.com/aws/agentcore-cli`) per their `SECURITY.md`
(private security advisory / `aws-security@amazon.com`, not a public issue,
since some of the underlying advisories touch auth/CORS/path-traversal
behavior). This section is the durable record in this repo in the meantime.

**Excluded GHSA IDs** (kept in sync with the `excluded_advisories` array in
`cicd/npm_audit_filtered.sh` — that script is the source of truth for *which*
IDs are excluded; this list is for humans reading this doc without opening
the script):

- `GHSA-g7r4-m6w7-qqqr` — esbuild dev-server arbitrary file read (Windows)
- `GHSA-xrhx-7g5j-rcj5`, `GHSA-3hrh-pfw6-9m5x`, `GHSA-f577-qrjj-4474`,
  `GHSA-2gcr-mfcq-wcc3`, `GHSA-wwfh-h76j-fc44`, `GHSA-j6c9-x7qj-28xf`,
  `GHSA-88fw-hqm2-52qc`, `GHSA-rv63-4mwf-qqc2`, `GHSA-wgpf-jwqj-8h8p` — hono
  (IP-restriction bypass, cookie injection, JWT auth-scheme confusion,
  `app.mount()` path decoding, `serve-static` path traversal, Lambda
  Set-Cookie merge, CORS wildcard-with-credentials, Body Limit bypass,
  Lambda@Edge header handling)
- `GHSA-q8mj-m7cp-5q26` — qs `stringify` DoS crash
- `GHSA-4x5r-pxfx-6jf8` — `@babel/core` arbitrary file read via
  `sourceMappingURL`
- `GHSA-8988-4f7v-96qf` — `@opentelemetry/core` unbounded memory in W3C
  Baggage propagation
- `GHSA-jxxr-4gwj-5jf2` — `brace-expansion` max-DoS-protection bypass
- `GHSA-gh4j-gqv2-49f6` — `fast-xml-parser` XMLBuilder comment/CDATA
  injection
- `GHSA-h67p-54hq-rp68` — `js-yaml` quadratic-complexity DoS in merge-key
  handling

**Re-check trigger.** Re-run `npm_audit_filtered.sh` (or raw `npm audit`)
against `infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/` after any
`@aws/agentcore` version bump. If a newer version resolves some/all of these
advisories, remove the now-fixed GHSA IDs from `excluded_advisories` in
`npm_audit_filtered.sh` and from the list above in the same change — don't
leave a stale exclusion for an advisory that no longer applies, since that
would silently mask a *future*, different advisory that happens to reuse the
same package name in npm's report.

## Clean scope

Non-destructive by default: `__pycache__`, `.pytest_cache`, `.ruff_cache`,
`.hypothesis`, `*.egg-info`, coverage output, `cdk.out`, compiled `.js`/`.d.ts`
build artifacts (except `jest.config.js`, which is source, not build output),
and `.ash/ash_output`. None of this requires reinstalling anything afterward.

`--deep` additionally removes `mcp/.venv`, `infrastructure/agentcore/.venv`, and both CDK
apps' `node_modules` — run `prep.sh` afterward before using the repo again.

## What this doesn't cover

- **Neither `deploy.sh` nor `destroy.sh` provisions AWS credentials** — you
  need a valid AWS session already configured (env vars, `~/.aws/credentials`,
  or IAM Identity Center) before running either.
- **Cost**: `deploy.sh` provisions a billed `trn1.2xlarge` instance among
  other resources. Check your EC2 quota and expected cost before running it
  against a real account — see the top-level `README.md` and
  `infrastructure/reward_server_cdk/README.md` for what gets created.
- **`security-check.sh` is a fast pre-deploy gate, not a full dependency/IaC
  review.** For a deeper STRIDE threat model or ASH-backed code
  review, run ASH separately in container mode (Docker, falling back to
  Podman) for stronger scanner isolation.
