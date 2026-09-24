# Running the NKI agent from the AWS Transform (ATX) CLI

The `atx` CLI (AWS Transform Custom) is one client of the `kernelforge-nki-mcp`
MCP server — the same durable contract the IDE plugin, Kiro, and AWS Batch use.
Unlike the IDE plugin (which installs via `/plugin install`), the `atx` CLI has
its **own** packaging model: a **transformation definition** plus an MCP-server
config in `~/.aws/atx/mcp.json`. There is no `atx plugin install`.

This directory holds the two ATX-native artifacts:

| Path | Role |
|---|---|
| `transformation-definition/transformation_definition.md` | The ATX transformation definition — the `atx` equivalent of `skills/nki-kernel/SKILL.md`. |
| `mcp.json` | The `~/.aws/atx/mcp.json` entry that registers `kernelforge-nki-mcp` for the `atx` CLI. |

## 1. Install the `atx` CLI

Install the AWS Transform CLI with the official installation script (requires
Node.js 22+ and Git):

```bash
curl -fsSL https://transform-cli.awsstatic.com/install.sh | bash
atx --version             # verify
```

The CLI authenticates with standard AWS credentials (environment variables or
`~/.aws/credentials`); attach the `AWSTransformCustomFullAccess` managed policy
to your IAM user or role. Full setup steps (supported regions, authentication
options) are in the
[AWS Transform Custom user guide](https://docs.aws.amazon.com/transform/latest/userguide/custom-get-started.html).

## 2. Register the MCP server for `atx`

The `atx` CLI reads MCP servers from `~/.aws/atx/mcp.json` (it does **not** read
this repo's `.mcp.json`, which is for IDE/Kiro clients). Copy the entry from
`atx/mcp.json` into `~/.aws/atx/mcp.json`, replacing the path placeholder with
the absolute path to this repo's `mcp/` directory:

```bash
mkdir -p ~/.aws/atx
REPO="$(cd "$(dirname "$0")/.." && pwd)"   # or the absolute path to this checkout
sed "s#REPLACE_WITH_ABSOLUTE_PATH#${REPO}#" atx/mcp.json > ~/.aws/atx/mcp.json

# confirm atx sees the server and its tools
atx mcp tools
atx mcp tools -s kernelforge-nki-mcp
```

> The MCP package is not on public PyPI, so the config launches the server from
> your local `mcp/` checkout via `uvx --from <repo>/mcp` — the same shape as the
> repo's `.mcp.json` and `.kiro/settings/mcp.json`. Keep the `--from`:
> a bare `uvx kernelforge-nki-mcp` would resolve the name from public PyPI,
> where it is not registered, so anyone who claims it could run code inside the agent process
> that launches the server, with its credentials and tool permissions. If the
> name is ever published, point `--from` at that trusted index explicitly
> (`uvx --index-url ... --from kernelforge-nki-mcp==<version>`) rather than
> falling back to bare-name resolution. Either way `uvx` honors the
> `mcp>=1.0.0,<2` pin in the package's own `pyproject.toml`, so the protocol
> version is bounded without restating it here.

The MCP server needs AWS credentials (standard chain) and the deployed
AgentCore runtime; set `AGENTCORE_ARN` / `AWS_REGION` in your environment (or in
the `env` block of the `mcp.json` entry) so `nki_generate_kernel` can reach the
runtime.

## 3. Run the migration

`atx` executes a **transformation definition** against a repository. Point it at
this repo's transformation definition and a target repo containing a kernel:

```bash
# Interactive — atx drives the transformation definition, calling the MCP tools
# (discover -> skill_lookup -> generate -> compile -> verify -> profile -> diff)
# with the two human gates (Scope, Requirements).
atx custom def exec \
  --code-repository-path ./examples/cuda_rmsnorm_migration \
  --configuration "additionalPlanContext=Follow the transformation definition in atx/transformation-definition/transformation_definition.md: convert the CUDA/PyTorch RMSNorm kernel to an @nki.jit NKI kernel and verify it on the Trn1 reward server."

# Non-interactive (CI / batch): add -x and -t.
atx custom def exec -x -t \
  --code-repository-path ./examples/cuda_rmsnorm_migration \
  --configuration "additionalPlanContext=..."
```

To publish this transformation into your account's registry (so teammates can
run it by name with `atx custom def exec -n <name>`):

```bash
atx custom def publish \
  --transformation-name pytorch-triton-to-nki \
  --source-directory atx/transformation-definition \
  --description "Convert PyTorch/Triton kernels to NKI, verified on Trainium."
# then:
atx custom def exec -n pytorch-triton-to-nki -p ./path/to/repo
```

> Publishing requires your own AWS account credentials. Definitions are
> account-scoped.
