# ATX NKI Agent — Solution Architecture

Maps every software component to (a) where it *runs* — developer workstation vs. AWS
— and (b) the exact file/directory in this repo that defines it. Two CDK apps deploy
the AWS side; everything else stays local. Everything that gets deployed to AWS lives
under `infrastructure/`; everything else in the repo is workstation-only.

```mermaid
flowchart TB
    subgraph WORKSTATION["💻 Developer workstation — nothing here is deployed to AWS"]
        direction TB

        subgraph IDE_LAYER["IDE / agent surface"]
            IDE["Kiro / Claude Code / Codex\n.kiro/steering/nki-kernel.md\n.claude-plugin/, .codex-plugin/"]
        end

        subgraph SKILL_LAYER["Skill (steering, loaded by the IDE)"]
            SKILL["skills/nki-kernel/SKILL.md\n+ references/*.md\nWorkflow + NKI hard rules"]
        end

        subgraph MCP_LAYER["MCP server — local process, stdio"]
            MCP["kernelforge-nki-mcp (in-repo, uvx --from <path>)\nmcp/src/kernelforge_nki_mcp/\n  server.py · discover.py · diff.py\n  clients.py · skill_db.py · config.py"]
        end

        subgraph DEVTOOL["Local dev tooling (never ships)"]
            CICD["cicd/*.sh + Makefile\nprep · lint · security-check\ndeploy.sh · destroy.sh"]
            SCRIPTS["scripts/*.py, nkibench/\nBenchmark/eval harness —\ncalls the deployed endpoints"]
        end

        REPO["Target git repo\n(the code being migrated)"]
    end

    subgraph AWS["☁️ AWS account — everything below is defined under infrastructure/\nand created by cdk deploy / agentcore deploy"]
        direction TB

        subgraph AGENTCORE_STACK["Stack 1: AgentCore Runtime\ndeployed by infrastructure/agentcore_cdk/atxnkiagent/ (agentcore create/deploy)"]
            direction TB
            AGENTCORE_APP["infrastructure/agentcore/app.py — Strands Agent\n@app.entrypoint handler\ninfrastructure/agentcore/router.py — ModelRouter"]
            AGENTCORE_L3["AgentCoreApplication L3 construct\n(@aws/agentcore-cdk, AWS-owned)\ninfrastructure/agentcore_cdk/.../lib/cdk-stack.ts"]
            AGENTCORE_RUNTIME["Bedrock AgentCore Runtime\n(managed compute — the actual\nAWS resource the L3 construct creates)"]
            AGENTCORE_ROLE["Execution IAM role\n(SigV4 identity for outbound calls)"]
            AGENTCORE_APP --> AGENTCORE_L3 --> AGENTCORE_RUNTIME
            AGENTCORE_RUNTIME -.-> AGENTCORE_ROLE
        end

        subgraph REWARD_STACK["Stack 2: TrainiumRewardServer\ndeployed by infrastructure/reward_server_cdk/ (cdk deploy)\ninfrastructure/reward_server_cdk/lib/reward-server-stack.ts"]
            direction TB

            subgraph VPC_BOX["VPC 10.42.0.0/16 — PRIVATE_ISOLATED, zero NAT"]
                direction LR
                CLIENTSG["AgentCoreClientSg\n(AgentCore runtime ENIs)"]
                TRN1["Trn1.2xlarge EC2 instance\ninfrastructure/reward_server/ (Flask, systemd, 4 workers)\n  server.py · auth.py (no app-layer auth by default)\n  compile.py · verify.py\n  profile.py · baseline.py\n  sandbox_env.py (subprocess env allowlist)"]
                ENDPOINTS["PrivateLink interface endpoints:\nBedrock Runtime, CloudWatch Logs,\nSSM(+Messages), EC2 Messages,\nCodeArtifact API+Repo\n+ S3 gateway endpoint\n(each with a scoped policy)"]
                CLIENTSG -->|"TCP 5050, plain HTTP\nno signing"| TRN1
                TRN1 -.->|"pip install at boot,\nvia CodeArtifact proxy"| ENDPOINTS
            end

            CODEARTIFACT["CodeArtifact domain + repo\n(nki-agent-pypi-proxy)\nproxies public PyPI"]
            S3ASSET["S3 asset bucket\n(reward_server/ source bundle,\nuploaded by cdk deploy)"]
            SSMPARAM["SSM Parameter\n/nki-agent/bootstrap-status"]
            ENDPOINTS -.-> CODEARTIFACT
            TRN1 -.->|"boot: fetch source"| S3ASSET
            TRN1 -.->|"boot: readiness marker"| SSMPARAM
        end

        BEDROCK["Amazon Bedrock\nClaude Opus 4.8 (default)\nBedrock CMI: Qwen3-Coder+SFT-v4 (not provisioned)"]
    end

    %% Cross-boundary edges (the actual runtime dataflow)
    IDE -->|"loads at session start"| SKILL
    SKILL -.->|"steers"| IDE
    IDE -->|"stdio, local process"| MCP
    MCP -->|"nki_generate_kernel\nSigV4, operator's own AWS creds\n(real: verified by AgentCore's own API)"| AGENTCORE_RUNTIME
    AGENTCORE_APP -->|"Bedrock Converse\nvia PrivateLink"| BEDROCK
    AGENTCORE_RUNTIME -->|"nki_compile / nki_verify / nki_profile\nplain HTTP, no signing — SG-to-SG only\n(no API Gateway)"| CLIENTSG
    MCP -.->|"nki_compile / nki_verify / nki_profile\n(direct leg) SigV4 computed but NOT verified\n(no-op today; needs a real\nnetwork path into the VPC)"| TRN1
    MCP -->|"nki_emit_diff\nlocal filesystem + git/gh CLI"| REPO
    SCRIPTS -.->|"benchmark calls,\nSigV4 computed but NOT verified\n(same no-op)"| TRN1
    CICD -->|"cdk deploy / agentcore deploy\n(provisions everything in AWS box)"| AGENTCORE_STACK
    CICD -->|"cdk deploy"| REWARD_STACK

    classDef workstation fill:#E8F4FD,stroke:#2E86C1,stroke-width:1px
    classDef aws fill:#FFF3E0,stroke:#E67E22,stroke-width:1px
    classDef awsmanaged fill:#FDEDEC,stroke:#C0392B,stroke-width:1px,stroke-dasharray: 4 2
    class IDE,SKILL,MCP,CICD,SCRIPTS,REPO workstation
    class AGENTCORE_APP,CLIENTSG,TRN1,ENDPOINTS,CODEARTIFACT,S3ASSET,SSMPARAM,AGENTCORE_ROLE aws
    class AGENTCORE_L3,AGENTCORE_RUNTIME,BEDROCK awsmanaged
```

## Legend

- **Blue** — runs on the developer's own machine. Nothing in this group is ever
  uploaded to AWS as infrastructure; the MCP server is *launched* locally (via `uvx`)
  even though it calls out to AWS services.
- **Orange** — infrastructure this repo's own code defines and CDK provisions, all
  under `infrastructure/` (`infrastructure/reward_server_cdk/`,
  `infrastructure/agentcore_cdk/`) — you can read every line of what gets created.
- **Red/dashed** — AWS-managed resources created *by* the orange layer's CDK code, but
  whose internals live in AWS-owned packages (`@aws/agentcore-cdk`'s L3 construct,
  Bedrock itself) — this repo configures them, doesn't implement them.
- **Dotted edges into `TRN1`** mark the MCP-direct calls whose SigV4 signature is
  computed but not verified by anything today — solid edges are either
  genuinely verified (AgentCore's own `InvokeAgentRuntime` API) or gated purely
  by network/security-group position, not by a signature.

## Component → deployment-target map

| Component | Runs on | Defined in | Deployed by |
|---|---|---|---|
| IDE/agent (Kiro, Claude Code, Codex) | Workstation | `.claude-plugin/`, `.codex-plugin/`, `.kiro/` | N/A — installed by the developer |
| Skill (steering doc) | Workstation (loaded into the IDE's context) | `skills/nki-kernel/SKILL.md` + `references/` | N/A — ships with the plugin/`.kiro/` |
| MCP server | Workstation (local process, `uvx`-launched) | `mcp/src/kernelforge_nki_mcp/` | `uvx --from <repo>/mcp kernelforge-nki-mcp` (in-repo, never bare-name/PyPI) — not CDK |
| `agentcore/app.py` (Strands agent code) | **AWS** — packaged and shipped into the AgentCore Runtime | `infrastructure/agentcore/app.py`, `infrastructure/agentcore/router.py` | `infrastructure/agentcore_cdk/atxnkiagent/` via `agentcore deploy` |
| AgentCore Runtime (managed compute) | **AWS** | AWS-owned (`@aws/agentcore-cdk`'s `AgentCoreApplication` L3 construct) | Same — `agentcore create`/`agentcore deploy` |
| AgentCore CDK wrapper stack | **AWS** (defines the stack; resources are AWS-managed) | `infrastructure/agentcore_cdk/atxnkiagent/agentcore/cdk/lib/cdk-stack.ts` | `agentcore deploy` (runs CDK under the hood) |
| Reward server app code | **AWS** — runs on the Trn1 instance | `infrastructure/reward_server/*.py` | `infrastructure/reward_server_cdk/` via `cdk deploy` (bundled as an S3 asset, installed at boot) |
| VPC, Trn1 instance, NLB, API Gateway, PrivateLink endpoints, CodeArtifact, S3 asset bucket | **AWS** | `infrastructure/reward_server_cdk/lib/reward-server-stack.ts`, `bin/app.ts`, `bootstrap/user-data.sh` | `cdk deploy` (via `cicd/deploy.sh`/`make deploy`) |
| Bedrock (Claude Opus 4.8, optional CMI) | **AWS**, AWS-managed service | N/A — configured via env vars/model IDs in `infrastructure/agentcore/app.py` | N/A — Bedrock itself is not deployed, only invoked |
| `cicd/*.sh`, `Makefile` | Workstation (or CI runner) | `cicd/` | N/A — these *drive* the two deploys above, they aren't deployed themselves |
| `scripts/*.py`, `nkibench/` (benchmark/eval harness) | Workstation | `scripts/`, `nkibench/` | N/A — calls the already-deployed reward server/AgentCore endpoints |
| Target repository being migrated | Workstation (or wherever `nki_emit_diff` points) | N/A — customer's own repo | N/A |

## `infrastructure/`: everything that gets deployed to AWS

```
infrastructure/
├── agentcore/            App code shipped into the AgentCore Runtime
├── agentcore_cdk/        CDK app that deploys the AgentCore Runtime
├── reward_server/        App code that runs on the Trn1 instance
└── reward_server_cdk/    CDK app that provisions the Trn1/VPC/API Gateway stack
```

Everything else in the repo — `mcp/`, `skills/`, `cicd/`, `scripts/`, `nkibench/`,
`.kiro/`, the plugin manifests — stays on the developer's workstation and is never
deployed as AWS infrastructure.

## The two CDK apps, and why there are two

- **`infrastructure/agentcore_cdk/atxnkiagent/`** — an `agentcore create`-scaffolded
  project. Its `agentcore/cdk/` subdirectory wraps AWS's own `@aws/agentcore-cdk` L3
  constructs; `agentcore deploy` synthesizes and deploys it. This is the AgentCore
  Runtime side.
- **`infrastructure/reward_server_cdk/`** — a hand-written CDK app (no
  `@aws/agentcore-cdk` dependency) that provisions the Trn1 reward server's full
  network stack from scratch: VPC, security groups, the EC2 instance itself, and
  every PrivateLink endpoint. Deployed with a plain `cdk deploy`. The AgentCore
  runtime reaches the reward server directly over SG-to-SG networking on port
  5050 (see `reward_server_cdk/README.md`); there is no NLB or API Gateway in
  this stack, since the security-group rule is the access control.

`cicd/deploy.sh` orchestrates both, AgentCore-first, then wires them together via
`infrastructure/reward_server_cdk/repoint-agentcore.sh` — see that script and
`cicd/README.md`'s "Deploy ordering" section for the exact sequencing and why
(AgentCore's REWARD_SERVER_URL / VPC network config isn't known until AFTER
the reward server exists, so AgentCore is deployed first and repointed at it
afterward, in place; access between the two stacks is gated by security
groups, with no IAM grant to sequence around).

There are, at the time of writing, two AgentCore deployment *paths* in this repo with
no single declared canonical one — `infrastructure/agentcore_cdk/atxnkiagent/` (used
by `cicd/deploy.sh` and shown above) and `infrastructure/agentcore/deploy.sh` (an
older, npm/pip-CLI-distinction-dependent fallback). Which one to keep is an open
decision.
