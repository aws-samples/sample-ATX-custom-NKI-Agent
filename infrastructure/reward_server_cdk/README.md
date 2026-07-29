# Reward Server — Infrastructure as Code

Provisions the Trn1 reward server **from scratch**, declaratively. No hand-built
AMIs, no manual SSH, no NAT Gateway: `cdk deploy` creates a small self-contained
VPC, launches a stock Neuron Deep Learning AMI into a private isolated subnet
with **no public IP**, and cloud-init installs the reward server from source
and runs it under systemd.

Anyone can reproduce the full setup from zero with the steps below — the extra
networking (VPC, PrivateLink endpoints, CodeArtifact PyPI proxy) is entirely
inside this stack; there is nothing extra for you to configure.

## What it creates

| Resource | Detail |
|---|---|
| VPC | `10.42.0.0/16`, private isolated subnets only, **zero NAT Gateways** |
| PrivateLink endpoints | S3 (gateway, free), Bedrock Runtime, CloudWatch Logs, SSM, SSM Messages, EC2 Messages, CodeArtifact API + Repositories, execute-api |
| CodeArtifact | Domain + repository with an external connection to public PyPI — the Trn1 installs `flask`/`gunicorn` through this proxy, never pypi.org directly |
| EC2 `trn1.2xlarge` | Stock Neuron DLAMI, IMDSv2, 200 GB gp3 (encrypted), **private subnet, no public IP** |
| Security group | Reward port 5050 accepts traffic **only from `AgentCoreClientSg`** (the AgentCore runtime's ENIs — direct SG-to-SG, no gateway in between; never `0.0.0.0/0`); SSH only if `sshCidr` is set (rarely needed — see Debugging) |
| IAM role (Trn1 instance) | Least-privilege: SSM (bootstrap-status marker + Session Manager), CloudWatch logs, CodeArtifact (read-only, scoped to this domain/repo) |
| systemd unit | `nki-reward.service` — 4 workers, request-scoped NEFF cache (no collision risk under concurrency), starts on boot, restarts on failure |

## Authentication: network isolation, no shared secret

There is no API Gateway, VPC Link, or NLB in front of the reward server —
that fronting layer was removed (it only ever added SigV4/IAM auth, which is
redundant given the network posture below, plus a 29-second integration
timeout that capped real compile/verify calls). The AgentCore runtime calls
the reward server **directly** at its private IP on port 5050
(`RewardUrlPrivate` stack output), over a security-group rule scoped to
exactly the AgentCore runtime's client ENIs (`AgentCoreClientSgId` stack
output) — see `reward-server-stack.ts`'s `RewardSg`/`AgentCoreClientSg`.
Since the Trn1 has no public IP and no route to the internet, the AgentCore
runtime's client security group is the *only* thing that can reach port 5050
at all. There is **no static shared secret for any caller to hold, rotate,
or leak**, and no `execute-api:Invoke` (or equivalent) IAM grant to manage —
trust is enforced entirely at the network layer.

MCP-direct calls (the MCP server's own `nki_compile`/`nki_verify`/
`nki_profile`/`nki_skill_lookup` tools, distinct from `nki_generate_kernel`'s
round-trip through AgentCore) need their own network path into this VPC —
there is no internet-reachable API Gateway mode to fall back to anymore. See
`mcp/src/kernelforge_nki_mcp/clients.py`'s `RewardServerClient` for the
current state of that path.

## Egress restriction on the Trn1's own security group (`restrictSubprocessEgress`)

Controls the Trn1 reward server's **outbound** traffic. Defaults to
`true` (restricted): the security group's egress is scoped to exactly this
VPC's PrivateLink endpoints (443) instead of `allowAllOutbound`. This closes
the gap where the reward server's handlers spawn a subprocess that runs
untrusted, LLM-generated kernel/reference source, and neither that
subprocess nor the running server has any legitimate reason to reach
anything outside this VPC's own endpoints.

```bash
npx cdk deploy                                    # restricted (default) — no flag needed
# or, to opt into unrestricted egress (e.g. for debugging):
npx cdk deploy -c restrictSubprocessEgress=false
REWARD_RESTRICT_EGRESS=false npx cdk deploy       # same, via env var (e.g. for CI)
```

## Deploy

```bash
cd infrastructure/reward_server_cdk
npm install
npx cdk bootstrap                      # once per account/region
npx cdk deploy                         # bundles reward_server/ as an S3 asset
```

This stack bootstraps into the account/region's default toolkit stack
(`CDKToolkit` / qualifier `hnb659fds`) rather than a dedicated one. A
dedicated qualifier was tried and dropped: the `agentcore` CLI's own internal
CDK deploys always target the default qualifier regardless of what any
project-local `cdk.json` declares, so pinning a custom qualifier for this app
only added a second, unused toolkit stack in the account instead of buying
real isolation. Standardizing on the default `CDKToolkit` means every CDK app
in this repo — this one, `agentcore_cdk/`, and anything `agentcore deploy`
provisions internally — shares a single bootstrap.

The reward server source is bundled as a CDK **S3 asset** and pulled by the
instance role at boot — no git credentials on the box, so it works even when the
source lives in a private repository. Outputs include `RewardUrlPrivate` (the direct
`http://<private-ip>:5050` endpoint AgentCore calls — set this as
`REWARD_SERVER_URL`) and `AgentCoreClientSgId` (attach the AgentCore runtime's
VPC-mode ENIs to this security group). There is no API Gateway invoke URL —
that fronting layer was removed; AgentCore reaches the reward server directly.

> **Migrating from a public-subnet deployment?** This revision moves the Trn1
> from the account's default VPC (public subnet) into this stack's own VPC
> (private isolated subnet). That is a new VPC/subnet/ENI for the instance, so
> the first `cdk deploy` after upgrading **replaces the running instance** —
> plan for capacity headroom (briefly two `trn1.2xlarge`) and re-run
> `repoint-agentcore.sh` afterwards since the private IP changes. Routine
> deploys after that (SG tweaks, endpoint additions) continue to update in
> place under the pinned logical ID, same as before.

## Point the AgentCore runtime at it (automated)

```bash
./repoint-agentcore.sh --region us-east-1
```

This reads the stack outputs, patches the AgentCore project's `agentcore.json`
(`REWARD_SERVER_URL` -> the reward server's direct private IP:5050, and removes any
leftover `REWARD_AUTH_TOKEN` env var), and runs an in-place `agentcore
deploy` — no manual editing. The runtime is still put in VPC mode using the
subnets this stack outputs (`VpcSubnets`) so it can reach the PrivateLink
endpoints (Bedrock, CloudWatch, SSM, CodeArtifact) — and reward-server calls
themselves go DIRECTLY to the Trn1 over SG-to-SG networking (no API Gateway,
no IAM grant to manage). See **Authentication** above.

## Run the benchmark

Once the runtime points at the live reward server:

```bash
cd ..
bash scripts/rerun_blog_results_opus48.sh    # needs REWARD_URL / AGENTCORE_ARN exported
```

## Debugging (no SSH needed)

The instance has no public IP, so SSH is rarely useful even when `sshCidr` is
set (there's still no route to it from outside the VPC). Use SSM Session
Manager instead — it works over the SSM/SSM Messages/EC2 Messages PrivateLink
endpoints this stack provisions, with no inbound rule at all:

```bash
aws ssm start-session --target <instance-id> --region us-east-1
```

## Tear down

```bash
cd infrastructure/reward_server_cdk && npx cdk destroy
```

The reward server is stateless (artifacts go to S3/CloudWatch), so destroy is safe.

## Security notes

- The reward server executes model-generated code. The instance has **no
  public IP and no route to the internet** — it lives in a private isolated
  subnet in a VPC this stack owns, with zero NAT Gateways. Every AWS service
  the instance needs (S3, SSM, CodeArtifact, Bedrock, CloudWatch Logs) is
  reached over AWS PrivateLink, never over the public internet.
- **Access control is network position, not IAM/SigV4.** There is no API
  Gateway in front of the reward server anymore — see Authentication above.
  Port 5050 accepts traffic only from `AgentCoreClientSg` (the AgentCore
  runtime's ENIs, SG-to-SG, direct); there is no other network path in and
  no static token for anyone to hold. MCP-direct callers without their own
  network path into this VPC currently have no working route to the reward
  server — this is an open item, not a
  documented, supported mode.
- **The Trn1 reward server's own security-group egress is deploy-time
  configurable** (`restrictSubprocessEgress` / `REWARD_RESTRICT_EGRESS`),
  and defaults to `true` (restricted). By default the SG's outbound rules
  are scoped to exactly this VPC's PrivateLink endpoints (443) — not
  `allowAllOutbound`. This closes the gap where the reward server's
  `compile`/`verify`/`profile`/`baseline` handlers spawn a subprocess that
  runs untrusted, LLM-generated kernel/reference source, and neither that
  subprocess nor the running server has any legitimate need to reach
  anything outside this VPC's own endpoints. Set
  `restrictSubprocessEgress=false` (or `REWARD_RESTRICT_EGRESS=false`) only to
  deliberately opt into unrestricted egress, e.g. for debugging a scenario
  that needs to reach something external directly from the box.
- `pip install` at boot resolves through a CodeArtifact repository (external
  connection to public PyPI) instead of pypi.org directly — the instance never
  needs internet egress for its dependencies.
- There is no shared-secret token anywhere in this stack — no SSM
  SecureString, nothing in the systemd unit, nothing in the CloudFormation
  template — access is gated entirely by network position (see Authentication
  above).
- SSH is disabled unless you explicitly pass `sshCidr` + `keyName`; use SSM Session
  Manager (enabled via the instance role, reachable with no public IP) for
  debugging instead.
