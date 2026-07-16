# kernelforge-nki-mcp

Stateless MCP server for the Kernel Forge NKI optimization agent. Exposes seven tools that an MCP-aware agent (Claude Code, Kiro, Codex) calls to discover candidate kernels, generate NKI source via AgentCore, and validate against a real Trainium device.

## Install

```bash
uv pip install kernelforge-nki-mcp
# or run directly:
uvx kernelforge-nki-mcp@latest
```

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
