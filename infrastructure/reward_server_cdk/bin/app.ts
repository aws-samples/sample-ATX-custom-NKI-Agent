#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { RewardServerStack } from '../lib/reward-server-stack';

const app = new cdk.App();

// Uses the account/region's default CDK bootstrap (qualifier "hnb659fds",
// stack "CDKToolkit") rather than a dedicated qualifier — deliberately, so
// this app shares the same toolkit stack as every other CDK app in the
// account (including the `agentcore` CLI's own internal deploys, which
// always bootstrap/target the default qualifier regardless of what any
// project-local cdk.json declares). A dedicated qualifier here bought no
// isolation in practice, since AgentCore never honored it, and just meant a
// second toolkit stack to keep in sync.

// All inputs come from context (cdk.json defaults or -c overrides) so the
// stack is reproducible from scratch with no code edits. The reward server
// source is bundled as an S3 asset by the stack — no git URL needed.
const instanceType = app.node.tryGetContext('instanceType') ?? 'trn1.2xlarge';
const sshCidr = app.node.tryGetContext('sshCidr'); // optional
const keyName = app.node.tryGetContext('keyName'); // optional
const rewardIngressCidrs = app.node.tryGetContext('rewardIngressCidrs'); // optional CSV
const availabilityZones = app.node.tryGetContext('availabilityZones'); // optional CSV
const agentCoreAzs = app.node.tryGetContext('agentCoreAzs'); // optional CSV

// Egress restriction for the Trn1 reward server's security group.
// Controls the SG's own outbound rules (restricted to this VPC's PrivateLink
// endpoints by default). See the `restrictSubprocessEgress` prop doc in
// reward-server-stack.ts for why the running server and its untrusted-code
// subprocess have no legitimate need for broader egress, and when you'd
// deliberately opt out (`-c restrictSubprocessEgress=false` or
// `REWARD_RESTRICT_EGRESS=false`) — e.g. temporarily debugging from the box
// with a need to reach something outside this VPC directly.
const restrictEgressContext = app.node.tryGetContext('restrictSubprocessEgress');
const restrictEgressEnv = process.env.REWARD_RESTRICT_EGRESS;
const restrictSubprocessEgress =
  restrictEgressContext !== undefined
    ? String(restrictEgressContext).toLowerCase() === 'true'
    : restrictEgressEnv !== undefined
      ? restrictEgressEnv.toLowerCase() === 'true'
      : true;

new RewardServerStack(app, 'TrainiumRewardServer', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
  },
  instanceType,
  sshCidr,
  keyName,
  availabilityZones: availabilityZones
    ? String(availabilityZones).split(',').map((s) => s.trim())
    : undefined,
  agentCoreAzs: agentCoreAzs
    ? String(agentCoreAzs).split(',').map((s) => s.trim())
    : undefined,
  rewardIngressCidrs: rewardIngressCidrs
    ? String(rewardIngressCidrs).split(',').map((s) => s.trim())
    : undefined,
  restrictSubprocessEgress,
});
