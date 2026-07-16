# Deployer network-condition IAM policy

`deployer-network-condition-policy.json` in this directory is a standalone
IAM policy document that enforces, at the IAM layer, that nobody can create
or update this project's AgentCore runtime with a network configuration
other than the one `infrastructure/reward_server_cdk` expects. Without this,
the reward server's *only* access control — `RewardSg` accepting port 5050
solely from `AgentCoreClientSg` — silently stops meaning anything if the
runtime is ever redeployed in `PUBLIC` mode or attached to a different
security group.

This document is **not a CDK construct, and not deployed by any stack
in this repo.** `infrastructure/agentcore_cdk/` is a thin wrapper around
AWS's own `@aws/agentcore-cdk` L3 construct and provisions no "deployer"
IAM identity of its own; `agentcore deploy` / `cicd/deploy.sh` run under
whatever ambient AWS credentials the operator or CI pipeline already has
(resolved at deploy time via `aws sts get-caller-identity`, never a fixed
account/role baked into this codebase). There is therefore no CDK resource
in this repo to attach this policy to — it must be applied directly, once,
to whichever real IAM user or role is authorized to run those deploy
commands, by whoever administers this AWS account's IAM.

## 1. Resolve the real security group ID

The policy's `bedrock-agentcore:securityGroups` condition needs the actual
`AgentCoreClientSg` ID from your deployed `reward_server_cdk` stack — it is
emitted as a stack output:

```bash
aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name TrainiumRewardServer \
  --query "Stacks[0].Outputs[?OutputKey=='AgentCoreClientSgId'].OutputValue | [0]" \
  --output text
```

(Same output `cicd/check-agentcore-network-drift.sh` reads to build its own
expected-config comparison — this policy and that script are meant to agree
on the same source of truth.)

## 2. Fill in the placeholder

Edit `deployer-network-condition-policy.json` and replace
`<AgentCoreClientSg-id>` with the value from step 1 (e.g. `sg-0123456789abcdef0`).
The file as checked in is **not valid to apply as-is** — the angle-bracket
placeholder is deliberate, so nobody copy-pastes a non-functional policy by
accident.

## 3. Apply it to the deployer identity

Attach as an inline policy on the IAM user or role that runs
`agentcore deploy` / `cicd/deploy.sh` (substitute the real principal name):

```bash
aws iam put-role-policy \
  --role-name <your-deployer-role-name> \
  --policy-name EnforceAgentCoreRuntimeNetworkConfig \
  --policy-document file://deployer-network-condition-policy.json
```

or, for an IAM user instead of a role:

```bash
aws iam put-user-policy \
  --user-name <your-deployer-user-name> \
  --policy-name EnforceAgentCoreRuntimeNetworkConfig \
  --policy-document file://deployer-network-condition-policy.json
```

Verified against [AWS Bedrock AgentCore — Use IAM condition keys with
AgentCore Runtime and built-in tools VPC
settings](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/security-vpc-condition.html):
these condition keys gate the *caller* of `CreateAgentRuntime`/
`UpdateAgentRuntime`, which is exactly the deployer identity this policy
targets — not the runtime's own execution role.

## 4. Re-verify with the existing detective control

After applying, `cicd/check-agentcore-network-drift.sh` (wired into
`cicd/deploy.sh` as an automatic post-deploy step, and available standalone
as `make check-network-drift`) still independently confirms the live
runtime's actual network config matches what `reward_server_cdk` expects.
The IAM policy above is the *preventive* half (stops the drift from being
possible for this principal); the drift check is the *detective* half
(confirms the live state after the fact, and catches drift caused by any
other principal this policy wasn't applied to). Both are recommended
together — this policy does not make the drift check redundant, since it
only constrains whichever single principal it's attached to.

## Applying this is outside this repository's own automation

This step is deliberately **not** run by any script in this repo (`deploy.sh`,
`make deploy`, etc.) — it is an account-administration action against IAM,
not a project resource, and applying it requires knowing the real deployer
identity and the real, currently-deployed security group ID, both of which
are environment-specific and not safe to infer or apply automatically.
