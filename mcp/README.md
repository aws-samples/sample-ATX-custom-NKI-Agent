# kernelforge-nki-mcp

Stateless MCP server for the Kernel Forge NKI optimization agent. Exposes seven tools that an MCP-aware agent (Claude Code, Kiro, Codex) calls to discover candidate kernels, generate NKI source via AgentCore, and validate against a real Trainium device.

## Install

This package is **not published to public PyPI**, and cannot be: its
`pyproject.toml` carries the `Private :: Do Not Upload` classifier, which PyPI
rejects on upload. Install or run it from this checkout, always naming the path:

```bash
uv pip install -e .          # from this directory
# or run directly, without installing:
uvx --from . kernelforge-nki-mcp
```

> Do not launch it as a bare `uvx kernelforge-nki-mcp`. `uvx` resolves bare names
> from public PyPI, and this name is unregistered there — anyone who claims it
> would get their code executed inside the agent process that launches the
> server, with its filesystem access, AWS credentials, environment and tool
> permissions. Every launcher config in this repo (`.mcp.json`,
> `.mcp.plugin.json`, `.kiro/settings/mcp.json`, `atx/mcp.json`) therefore passes
> `--from <path>`.

> **Stale-cache trap when iterating on this package.** `uvx --from <local path>`
> caches the wheel it builds, and will keep serving it after you edit the source.
> Prefix with `UV_NO_CACHE=1` (or `uv run --project mcp`) when you need your
> changes to actually take effect.

## Configure

```bash
# The deployed AgentCore runtime, addressed by ARN (not a base URL — the
# data-plane path is /runtimes/{arn}/invocations?qualifier=...):
export AGENTCORE_ARN="$(aws bedrock-agentcore-control list-agent-runtimes \
  --region us-east-1 \
  --query "agentRuntimes[?agentRuntimeName=='atxnkiagent_nki_agent'].agentRuntimeArn" \
  --output text)"
export AGENTCORE_QUALIFIER="DEFAULT"          # optional; DEFAULT if unset
export REWARD_SERVER_URL="http://<trn1-host>:5050"
export AWS_REGION="us-east-1"
# AWS credentials via standard chain (env, ~/.aws/credentials, IAM Identity Center)
```

`nki_generate_kernel` is the only tool that needs `AGENTCORE_ARN`; the discovery,
skill-lookup and diff tools are local and work without it. `REWARD_SERVER_URL`
is only reachable from inside the reward server's VPC — calling `nki_compile` /
`nki_verify` / `nki_profile` directly from a workstation needs a network path in
(VPN / Direct Connect / SSM tunnel). The normal flow goes through
`nki_generate_kernel`, and the AgentCore runtime talks to the reward server for
you.

## Tools

| Tool | Purpose |
|---|---|
| `nki_discover_kernels` | Walk a repo, find `@triton.jit` and hot `nn.Module.forward` |
| `nki_generate_kernel`  | Multi-turn generation via AgentCore (Opus 4.8 / Qwen3-SFT-v4 / vLLM) |
| `nki_compile`          | `neuronx-cc` compile via reward server |
| `nki_verify`           | torch.allclose on real Trainium |
| `nki_profile`          | `neuron-profile` latency + utilization |
| `nki_skill_lookup`     | Search NKI patterns / examples |
| `nki_emit_diff`        | Generate patch / open PR |

See `src/kernelforge_nki_mcp/server.py` for the canonical schemas.

## Auth

The server is a customer-side process. It uses the standard AWS credential chain to SigV4-sign requests to AgentCore and the Trn1 reward server. No long-lived credentials are stored.

## License

MIT-0. See the repository LICENSE file.
