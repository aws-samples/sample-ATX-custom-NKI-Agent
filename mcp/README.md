# kernelforge-nki-mcp

Stateless MCP server for the Kernel Forge NKI optimization agent. Exposes seven tools that an MCP-aware agent (Claude Code, Kiro, Codex) calls to discover candidate kernels, generate NKI source via AgentCore, and validate against a real Trainium device.

## Install

This package is **not published to public PyPI**. Install or run it from this
checkout, always naming the path:

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
> `.kiro/settings/mcp.json`, `atx/mcp.json`) therefore passes `--from <path>`.

## Configure

```bash
export AGENTCORE_ENDPOINT="https://<your-agentcore-endpoint>"
export REWARD_SERVER_URL="http://<trn1-host>:5050"
export AWS_REGION="us-east-1"
# AWS credentials via standard chain (env, ~/.aws/credentials, IAM Identity Center)
```

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
