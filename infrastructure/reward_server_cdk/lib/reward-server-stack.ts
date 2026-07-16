import {
  CfnOutput,
  Duration,
  Stack,
  type StackProps,
  Tags,
} from 'aws-cdk-lib';
import * as codeartifact from 'aws-cdk-lib/aws-codeartifact';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import { Construct } from 'constructs';
import * as fs from 'fs';
import * as path from 'path';

export interface RewardServerStackProps extends StackProps {
  /**
   * Trn1 instance size. Default trn1.2xlarge (1 NeuronCore-v2 device pair) —
   * enough for the reward server's 4-worker gunicorn pool.
   */
  readonly instanceType?: string;
  /**
   * Availability zone the Trn1 (and its private subnet) is placed in. Trn1 is
   * only offered in some AZs (e.g. us-east-1c / us-east-1f). Pinned to a
   * SINGLE AZ on purpose: this stack now owns its own VPC (see below), so
   * there is no default-VPC subnet-ordering drift risk, but keeping a single
   * AZ still avoids spreading interface endpoints (billed per-AZ) across AZs
   * the runtime doesn't need. Default: us-east-1f (matches the
   * currently-deployed instance).
   */
  readonly availabilityZones?: string[];
  /**
   * AZs the AgentCore runtime's VPC-mode ENIs may use. AgentCore VPC mode
   * supports only a subset of AZs. The runtime now shares this stack's own
   * VPC (private, isolated subnets) rather than a default VPC's public ones.
   * Default: us-east-1a / 1b / 1d.
   */
  readonly agentCoreAzs?: string[];
  /**
   * Extra CIDRs allowed to reach the reward port (5050), on top of the
   * least-privilege client-SG rule. Default: none — only the AgentCore
   * runtime's client SG may reach 5050. NEVER pass 0.0.0.0/0 here — the reward
   * server runs model-generated code, and the instance has no public IP to
   * begin with, so a CIDR here only makes sense as a VPN/Direct-Connect range
   * that already terminates inside this VPC. Set only for local debugging.
   */
  readonly rewardIngressCidrs?: string[];
  /**
   * Optional CIDR allowed SSH (22) for debugging. Omit to disable SSH
   * entirely (recommended — the instance has no public IP, so SSH is only
   * reachable from inside the VPC anyway; prefer SSM Session Manager, which
   * needs no inbound rule at all).
   */
  readonly sshCidr?: string;
  /** Existing EC2 key pair name for SSH. Omit if sshCidr is unset. */
  readonly keyName?: string;
  /**
   * Restricts the Trn1 reward server's security-group egress to exactly the
   * PrivateLink/gateway endpoints this stack provisions (443 to `endpointSg`,
   * plus 443 to the S3 prefix list for the gateway endpoint) instead of the
   * unrestricted `allowAllOutbound` default a plain CDK security group would
   * otherwise get. Default: `true` (restricted).
   *
   * This closes the gap where the compile/verify/profile/baseline handlers spawn a
   * subprocess that executes untrusted, LLM-generated kernel/reference source
   * (see `reward_server/sandbox_env.py`'s docstring for the companion fix
   * on that same subprocess's environment). Neither the running server
   * nor that subprocess has any legitimate reason to reach anything outside
   * this VPC's own PrivateLink endpoints — the AWS calls at boot
   * (S3/CodeArtifact/SSM, in `user-data.sh`) are the only outbound traffic
   * this design needs, and those already go through the endpoints below.
   * There is no NAT/internet gateway on this VPC either way (see `natGateways:
   * 0`), so this restricts traffic within the VPC, not to the internet — the
   * egress-restricted SG is an additional, narrower boundary on top of that.
   *
   * Checked via `-c restrictSubprocessEgress=true|false` or
   * `REWARD_RESTRICT_EGRESS=true|false` — see `bin/app.ts`. Set to `false`
   * only for debugging a scenario that legitimately needs broader egress
   * (e.g. temporarily reaching an external endpoint from the box directly).
   */
  readonly restrictSubprocessEgress?: boolean;
  /**
   * When true, editing the bootstrap (user-data) forces a full instance
   * replacement so the change re-runs from a clean first boot. Default: false.
   *
   * Left off by default on purpose: user-data only runs at first boot, so a
   * bootstrap edit otherwise has no effect on a running box — but replacement
   * creates a new trn1 before deleting the old, needing headroom for two
   * trn1.2xlarge at once (scarce, quota-limited). Set this true only for a
   * deliberate reprovision when you have that capacity; leave it false for
   * routine deploys (endpoints, SG tweaks) so a healthy GPU box is never torn
   * down by surprise.
   */
  readonly reprovisionOnBootstrapChange?: boolean;
}

/**
 * Clean, from-scratch provisioning for the Trn1 reward server.
 *
 * Everything is declarative: a stock Neuron Deep Learning AMI is launched into
 * a private, isolated subnet with no public IP, and cloud-init user-data
 * installs the reward server from source and runs it under systemd. No
 * hand-built AMIs, no manual SSH steps, no NAT Gateway — `cdk deploy`
 * reproduces the whole thing from zero using only this stack's own VPC and
 * AWS PrivateLink.
 *
 * Why not the default VPC: the previous revision of this stack used
 * `Vpc.fromLookup({ isDefault: true })` and placed the Trn1 in a PUBLIC
 * subnet purely for demo convenience (no extra networking to reproduce the
 * blog). But the reward server's entire job is to execute model-generated
 * code, and a public subnet means the security group is the *only* control
 * standing between that code-execution endpoint and the internet — one
 * manual SG edit, one additional attached SG, or one future port opened
 * carelessly is enough to expose it. This stack instead owns a minimal VPC
 * with zero NAT Gateways and reaches every AWS service it needs (S3, SSM,
 * CodeArtifact, Bedrock, CloudWatch Logs) over PrivateLink, so the instance
 * has no public IP and no route to the internet at all — an SG mistake alone
 * can no longer make it internet-reachable.
 *
 * The auth token is generated on the instance at first boot and stored in SSM
 * Parameter Store (SecureString) so nothing secret is baked into the template.
 */
export class RewardServerStack extends Stack {
  constructor(scope: Construct, id: string, props: RewardServerStackProps) {
    super(scope, id, props);

    const instanceTypeStr = props.instanceType ?? 'trn1.2xlarge';

    // Trn1 is only offered in some AZs; pin to a single AZ so the VPC, the
    // instance, and the PrivateLink endpoints all agree on where the reward
    // server lives. us-east-1f matches the currently-deployed instance.
    const trnAzs = props.availabilityZones ?? ['us-east-1f'];
    const primaryAz = trnAzs[0];

    // AgentCore VPC mode places the runtime's ENIs only in a subset of AZs
    // (not necessarily the Trn1 AZ). The runtime and the Trn1 now share this
    // stack's own VPC, so both sets of subnets live in the same VPC and can
    // route to each other without any public IP on either side.
    const agentCoreAzs = props.agentCoreAzs ?? ['us-east-1a', 'us-east-1b', 'us-east-1d'];

    // ── Self-contained VPC, no NAT Gateway ────────────────────────────────
    // A single small VPC this stack owns end-to-end (vs. depending on the
    // account's default VPC). Every subnet is PRIVATE_ISOLATED: no route to
    // an internet gateway or a NAT gateway. All AWS-service reachability
    // comes from the PrivateLink endpoints below, and S3 access comes from a
    // gateway endpoint (also no internet path, and no hourly charge).
    // AZs cover both the Trn1's AZ and every AgentCore runtime AZ so the
    // runtime's ENIs and the Trn1 end up in the same VPC regardless of which
    // AZ each needs.
    const vpcAzs = Array.from(new Set([...trnAzs, ...agentCoreAzs]));
    const vpc = new ec2.Vpc(this, 'Vpc', {
      ipAddresses: ec2.IpAddresses.cidr('10.42.0.0/16'),
      availabilityZones: vpcAzs,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: 'private-isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
    });

    const trnSubnets = vpc
      .selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_ISOLATED })
      .subnets.filter((s) => trnAzs.includes(s.availabilityZone));
    const agentCoreSubnets = vpc
      .selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_ISOLATED })
      .subnets.filter((s) => agentCoreAzs.includes(s.availabilityZone));

    // Bundle the reward server as an S3 asset. No git credentials on the box —
    // the instance role pulls exactly the source that was synthesized. Works
    // for private repos (like code.aws.dev) that a fresh EC2 can't clone.
    const rewardAsset = new s3assets.Asset(this, 'RewardServerSource', {
      path: path.join(__dirname, '..', '..', 'reward_server'),
    });

    // Reward port is locked to the AgentCore runtime's client SG by default
    // (added below) — least privilege. Extra CIDRs are opt-in only, for local
    // debugging from something that already has private connectivity into
    // this VPC (VPN / Direct Connect / peering) — there is no public IP path.
    const rewardCidrs = props.rewardIngressCidrs ?? [];

    // The reward server and its compile/verify/profile/baseline
    // subprocesses (which execute untrusted, LLM-generated source) have no
    // legitimate reason to reach anything outside this VPC's own PrivateLink
    // endpoints. Restricted by default; see the `restrictSubprocessEgress`
    // prop doc for the full rationale and the opt-out switch.
    const restrictEgress = props.restrictSubprocessEgress ?? true;

    const sg = new ec2.SecurityGroup(this, 'RewardSg', {
      vpc,
      description: 'Trn1 reward server - private subnet, no public IP; port 5050 locked to AgentCore client SG',
      allowAllOutbound: !restrictEgress,
    });
    for (const cidr of rewardCidrs) {
      if (cidr === '0.0.0.0/0') {
        throw new Error(
          'refusing to open reward port 5050 to 0.0.0.0/0 — the reward server ' +
            'runs model-generated code; scope rewardIngressCidrs to the caller.',
        );
      }
      sg.addIngressRule(ec2.Peer.ipv4(cidr), ec2.Port.tcp(5050), `reward ${cidr}`);
    }
    if (props.sshCidr) {
      sg.addIngressRule(ec2.Peer.ipv4(props.sshCidr), ec2.Port.tcp(22), 'ssh (debug)');
    }

    // Dedicated client SG for the AgentCore runtime's VPC-mode ENIs. The
    // runtime calls the reward server DIRECTLY at its private IP:5050 over this
    // SG-to-SG rule (see the ingress rule just below) — there is no API Gateway
    // / VPC Link / NLB in the path anymore. That fronting stack was removed
    // because its only security contribution was SigV4/IAM auth, which is
    // redundant here: the Trn1 lives in a PRIVATE_ISOLATED subnet with no public
    // IP and no internet route, and its RewardSg accepts 5050 from THIS client
    // SG only — so the single AgentCore runtime is the only thing that can even
    // reach the port. Removing the gateway also removes its 29s integration
    // timeout, which capped compile/verify calls that legitimately run minutes.
    // This SG is also the source for the VPC-endpoint ingress below (Bedrock,
    // CloudWatch Logs) that the runtime still needs for model calls + logging.
    const clientSg = new ec2.SecurityGroup(this, 'AgentCoreClientSg', {
      vpc,
      description: 'AgentCore runtime ENIs - direct client of the reward server on 5050',
      allowAllOutbound: true,
    });
    // The reward server accepts 5050 from the AgentCore runtime's ENIs only.
    sg.addIngressRule(clientSg, ec2.Port.tcp(5050), 'reward from the AgentCore runtime ENIs (direct, no gateway)');

    // ── Runtime + Trn1 egress to AWS APIs, without a NAT or internet gateway ──
    // Neither the AgentCore runtime's ENIs nor the Trn1 instance have a public
    // IP or a route to the internet. Everything they need — Bedrock (model
    // calls), CloudWatch Logs, SSM (Session Manager + the token param), and
    // CodeArtifact (pip installs at boot, see below) — is reached via
    // PrivateLink interface endpoints, plus a free S3 gateway endpoint for the
    // reward-server source asset. Traffic never leaves the AWS network.
    //
    // Dedicated SG for the endpoint ENIs: reachable only from the runtime's
    // client SG and the reward SG, only on 443. No CIDR-wide access.
    const endpointSg = new ec2.SecurityGroup(this, 'ApiEndpointSg', {
      vpc,
      description: 'PrivateLink endpoints - 443 from AgentCore client SG and the Trn1 reward SG only',
      allowAllOutbound: true,
    });
    endpointSg.addIngressRule(clientSg, ec2.Port.tcp(443), 'HTTPS from AgentCore runtime ENIs');
    endpointSg.addIngressRule(sg, ec2.Port.tcp(443), 'HTTPS from the Trn1 reward server (bootstrap)');

    // Now that
    // endpointSg exists, grant the Trn1 reward server's SG egress to exactly
    // this — the PrivateLink interface endpoints below (Bedrock, CloudWatch
    // Logs, SSM, SSM Messages, EC2 Messages, CodeArtifact API+repo,
    // execute-api) — plus the S3 gateway endpoint further below, and nothing
    // else. Skipped entirely when `restrictSubprocessEgress=false`, in which
    // case `allowAllOutbound: true` (set above) already covers everything.
    if (restrictEgress) {
      sg.addEgressRule(
        endpointSg,
        ec2.Port.tcp(443),
        'HTTPS to this VPCs PrivateLink endpoints only (no broader egress for the untrusted-code subprocess)',
      );
    }

    // Endpoints live in every subnet that has a client (both the AgentCore
    // runtime's AZs and the Trn1's AZ) so every ENI has a same-AZ endpoint.
    const endpointSubnets: ec2.SubnetSelection = {
      subnets: Array.from(new Set([...agentCoreSubnets, ...trnSubnets])),
    };

    // S3 gateway endpoint: free, no hourly charge. Used by the Trn1 to fetch
    // the reward-server source asset at boot with no internet path. Scoped
    // (via addToPolicy below) to read-only actions on exactly that
    // asset's bucket (the same bucket `rewardAsset.grantRead(role)` already
    // grants IAM access to below) — narrower than S3's default full-access
    // endpoint policy.
    const s3Endpoint = vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // S3 gateway endpoints are reached via a route-table entry + the AWS-
    // managed prefix list for the service, not an ENI/security-group — so
    // RewardSg's egress rule for S3 (needed for the restricted-egress
    // mode above) must reference that prefix list rather than endpointSg.
    if (restrictEgress) {
      const s3PrefixList = ec2.PrefixList.fromLookup(this, 'S3PrefixList', {
        prefixListName: `com.amazonaws.${Stack.of(this).region}.s3`,
      });
      sg.addEgressRule(
        ec2.Peer.prefixList(s3PrefixList.prefixListId),
        ec2.Port.tcp(443),
        'HTTPS to the S3 gateway endpoint only',
      );
    }

    // Every interface/gateway endpoint below gets an explicit,
    // scoped policy via `addToPolicy()` instead of AWS's default full-access
    // policy (`{Effect:"Allow",Principal:"*",Action:"*",Resource:"*"}`,
    // applied to any endpoint with no custom policy). This matches AWS's own
    // documented pattern for private APIs/PrivateLink — endpoint policies
    // are additive to, not a substitute for, the IAM policies already
    // scoping `role` above; both must independently allow an action for it
    // to succeed. Each policy below is scoped to the exact actions/resources
    // this design's IAM role or the AgentCore/reward-server calls actually
    // use — see the comment on each endpoint for the specific
    // justification. Principal is deliberately `AnyPrincipal` (AWS's own
    // recommended pattern: the resource ARNs constrain *what*, IAM identity
    // policies constrain *who* — the endpoint's own SG, restricted to
    // `clientSg`/`sg` above, is what constrains network reachability to
    // these ARNs in the first place). Every `addToPolicy` call below must
    // supply a Principal (CDK's own requirement for VPC endpoint policy
    // statements).
    const anyPrincipal = new iam.AnyPrincipal();

    s3Endpoint.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [anyPrincipal],
        actions: ['s3:GetObject', 's3:GetBucketLocation', 's3:ListBucket'],
        resources: [
          `arn:aws:s3:::${rewardAsset.s3BucketName}`,
          `arn:aws:s3:::${rewardAsset.s3BucketName}/*`,
        ],
      }),
    );

    // AgentCore's own zip-code-artifact bucket. The AgentCore runtime
    // (infrastructure/agentcore_cdk/) is deployed in this VPC's own
    // network mode with no NAT gateway, and — being a direct-code-deploy
    // (zip-based) agent, not a container/ECR agent — fetches its code from
    // an AWS-internal, per-region bucket pattern (documented at
    // https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html#agentcore-vpc-endpoints).
    // Without this statement, that fetch has no path (no NAT, and the S3
    // gateway endpoint policy above was scoped only to the reward server's
    // own asset bucket under the scoped policy above) — the runtime hangs trying to reach
    // it and is killed by AgentCore's 30s init timeout before it ever gets
    // far enough to write a single log line. Confirmed live: identical
    // "Runtime initialization time exceeded... 30s" failure with zero
    // CloudWatch log bytes, reproduced with both a real task payload and a
    // minimal one that never reaches app.py's handler() body — i.e. the
    // failure is in the runtime's own cold start, before any application
    // code runs, and is unrelated to this stack's own API Gateway mode.
    s3Endpoint.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [anyPrincipal],
        actions: ['s3:GetObject'],
        resources: [
          `arn:aws:s3:::acr-code-*-${Stack.of(this).region}-an`,
          `arn:aws:s3:::acr-code-*-${Stack.of(this).region}-an/*`,
        ],
      }),
    );

    // Bedrock runtime: the agent calls invoke_model/converse here for the
    // Opus model (agentcore/app.py). Scoped to exactly the actions used.
    const bedrockRuntimeEndpoint = new ec2.InterfaceVpcEndpoint(this, 'BedrockRuntimeEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    bedrockRuntimeEndpoint.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [anyPrincipal],
        actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream', 'bedrock:Converse', 'bedrock:ConverseStream'],
        resources: ['*'],
      }),
    );

    // CloudWatch Logs: so the container / instance can actually emit logs
    // from in-VPC. Scoped to the log-stream write actions the CloudWatch
    // agent / any log producer on this instance needs, not full CloudWatch
    // access.
    const cloudWatchLogsEndpoint = new ec2.InterfaceVpcEndpoint(this, 'CloudWatchLogsEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    cloudWatchLogsEndpoint.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [anyPrincipal],
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          'logs:DescribeLogGroups',
          'logs:DescribeLogStreams',
        ],
        resources: [`arn:aws:logs:${this.region}:${this.account}:*`],
      }),
    );

    // SSM + SSM Messages + EC2 Messages: required together for Session
    // Manager to work with no inbound rule and no public IP — this is the
    // debugging path that replaces SSH now that the instance has no public
    // IP. Also used for the bootstrap-status parameter. Scoped to the
    // Session-Manager/agent actions SSM needs plus the exact
    // GetParameter/PutParameter grant `role` already has on `/nki-agent/*`
    // (the endpoint policy is additive to that IAM scoping, not a
    // replacement for it).
    const applySsmPolicy = (endpoint: ec2.InterfaceVpcEndpoint) => {
      endpoint.addToPolicy(
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          principals: [anyPrincipal],
          actions: [
            'ssmmessages:CreateControlChannel',
            'ssmmessages:CreateDataChannel',
            'ssmmessages:OpenControlChannel',
            'ssmmessages:OpenDataChannel',
            'ec2messages:AcknowledgeMessage',
            'ec2messages:DeleteMessage',
            'ec2messages:FailMessage',
            'ec2messages:GetEndpoint',
            'ec2messages:GetMessages',
            'ec2messages:SendReply',
            'ssm:UpdateInstanceInformation',
            'ssm:ListInstanceAssociations',
            'ssm:DescribeInstanceProperties',
            'ssm:DescribeDocumentParameters',
          ],
          resources: ['*'],
        }),
      );
      endpoint.addToPolicy(
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          principals: [anyPrincipal],
          actions: ['ssm:GetParameter', 'ssm:PutParameter'],
          resources: [`arn:aws:ssm:${this.region}:${this.account}:parameter/nki-agent/*`],
        }),
      );
    };
    const ssmEndpoint = new ec2.InterfaceVpcEndpoint(this, 'SsmEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.SSM,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    applySsmPolicy(ssmEndpoint);
    const ssmMessagesEndpoint = new ec2.InterfaceVpcEndpoint(this, 'SsmMessagesEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.SSM_MESSAGES,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    applySsmPolicy(ssmMessagesEndpoint);
    const ec2MessagesEndpoint = new ec2.InterfaceVpcEndpoint(this, 'Ec2MessagesEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.EC2_MESSAGES,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    applySsmPolicy(ec2MessagesEndpoint);

    // CodeArtifact API + repository endpoints: the Trn1 authenticates pip
    // against a CodeArtifact repository (see below) instead of the public
    // PyPI index, entirely over PrivateLink. Scoped to exactly the read-only
    // actions `role` is granted below (GetAuthorizationToken,
    // GetRepositoryEndpoint, ReadFromRepository) — no publish/delete. Also
    // includes `sts:GetServiceBearerToken`: CodeArtifact's GetAuthorizationToken
    // API evaluates this as part of the same request (AWS's own VPC-endpoint
    // policy example for CodeArtifact lists it in this same statement, not on
    // a separate STS endpoint/policy — `aws codeartifact login` fails with
    // "no VPC endpoint policy allows the sts:GetServiceBearerToken action"
    // otherwise, even though the IAM role itself is separately granted this
    // action too (see role.addToPolicy for the same action, below) — the
    // identity policy and this endpoint policy are both required,
    // independently, per AWS's endpoint-policy model).
    const applyCodeArtifactPolicy = (endpoint: ec2.InterfaceVpcEndpoint) => {
      endpoint.addToPolicy(
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          principals: [anyPrincipal],
          actions: [
            'codeartifact:GetAuthorizationToken',
            'codeartifact:GetRepositoryEndpoint',
            'codeartifact:ReadFromRepository',
            'sts:GetServiceBearerToken',
          ],
          resources: ['*'],
        }),
      );
    };
    const codeArtifactApiEndpoint = new ec2.InterfaceVpcEndpoint(this, 'CodeArtifactApiEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.CODEARTIFACT_API,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    applyCodeArtifactPolicy(codeArtifactApiEndpoint);
    const codeArtifactRepositoriesEndpoint = new ec2.InterfaceVpcEndpoint(this, 'CodeArtifactRepositoriesEndpoint', {
      vpc,
      service: ec2.InterfaceVpcEndpointAwsService.CODEARTIFACT_REPOSITORIES,
      subnets: endpointSubnets,
      securityGroups: [endpointSg],
      privateDnsEnabled: true,
    });
    applyCodeArtifactPolicy(codeArtifactRepositoriesEndpoint);

    // (No execute-api VPC endpoint: the runtime no longer calls an API Gateway
    // — it reaches the reward server directly on 5050 within this VPC.)

    // ── CodeArtifact: private PyPI proxy ──────────────────────────────────
    // The Trn1 has no route to the public internet, so `pip install` at boot
    // can't reach pypi.org directly. A CodeArtifact repository with an
    // external connection to the public PyPI repository proxies and caches
    // packages: the first pip install for a given package/version pulls
    // through to real PyPI (CodeArtifact's own managed egress, not the
    // instance's), and every subsequent install (future re-bootstraps,
    // instance replacement) is served from the cache with no external call
    // at all. Reached from the instance purely over the endpoints above.
    const codeArtifactDomain = new codeartifact.CfnDomain(this, 'CodeArtifactDomain', {
      domainName: 'nki-agent',
    });
    const codeArtifactRepo = new codeartifact.CfnRepository(this, 'CodeArtifactRepo', {
      domainName: codeArtifactDomain.domainName,
      repositoryName: 'nki-agent-pypi-proxy',
      description: 'Proxies public PyPI for the reward server bootstrap (flask, gunicorn) - no internet egress from the Trn1.',
      externalConnections: ['public:pypi'],
    });
    codeArtifactRepo.addDependency(codeArtifactDomain);

    // Least-privilege role: SSM (bootstrap-status marker + Session Manager) +
    // CloudWatch logs + CodeArtifact (read-only, scoped to this domain/repo)
    // for the bootstrap's pip install.
    const statusParamName = `/nki-agent/bootstrap-status`;
    const role = new iam.Role(this, 'RewardRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy'),
      ],
    });
    // Scope to the /nki-agent/* prefix so the instance can write the
    // bootstrap-status marker, but nothing else.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['ssm:PutParameter', 'ssm:GetParameter'],
        resources: [
          `arn:aws:ssm:${this.region}:${this.account}:parameter/nki-agent/*`,
        ],
      }),
    );
    // Scoped to exactly this domain/repository — read-only (auth token +
    // package fetch), no publish/delete permissions.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['codeartifact:GetAuthorizationToken'],
        resources: [
          `arn:aws:codeartifact:${this.region}:${this.account}:domain/${codeArtifactDomain.domainName}`,
        ],
      }),
    );
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['codeartifact:GetRepositoryEndpoint', 'codeartifact:ReadFromRepository'],
        resources: [
          `arn:aws:codeartifact:${this.region}:${this.account}:repository/${codeArtifactDomain.domainName}/${codeArtifactRepo.repositoryName}`,
        ],
      }),
    );
    // GetServiceBearerToken is an STS action required by `aws codeartifact
    // login` regardless of which domain/repo it targets.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['sts:GetServiceBearerToken'],
        resources: ['*'],
        conditions: {
          StringEquals: { 'sts:AWSServiceName': 'codeartifact.amazonaws.com' },
        },
      }),
    );
    // Let the instance download the bundled source asset.
    rewardAsset.grantRead(role);

    // Stock Neuron DLAMI — looked up at synth time, never hand-built.
    const machineImage = ec2.MachineImage.lookup({
      name: 'Deep Learning AMI Neuron (Ubuntu 22.04)*',
      owners: ['amazon'],
    });

    // Cloud-init: fetch the bundled source from S3, authenticate pip against
    // CodeArtifact, run under systemd so the server survives reboots and
    // starts on boot, and record bootstrap outcome to SSM for repoint-agentcore.sh.
    const bootstrap = fs.readFileSync(
      path.join(__dirname, '..', 'bootstrap', 'user-data.sh'),
      'utf8',
    );
    const userData = ec2.UserData.custom(
      bootstrap
        .replace(/__ASSET_BUCKET__/g, rewardAsset.s3BucketName)
        .replace(/__ASSET_KEY__/g, rewardAsset.s3ObjectKey)
        .replace(/__STATUS_PARAM__/g, statusParamName)
        .replace(/__REGION__/g, this.region)
        .replace(/__CODEARTIFACT_DOMAIN__/g, codeArtifactDomain.domainName)
        .replace(/__CODEARTIFACT_DOMAIN_OWNER__/g, this.account)
        .replace(/__CODEARTIFACT_REPO__/g, codeArtifactRepo.repositoryName as string),
    );

    const keyPair = props.keyName
      ? ec2.KeyPair.fromKeyPairName(this, 'KeyPair', props.keyName)
      : undefined;

    const instance = new ec2.Instance(this, 'RewardServer', {
      vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
        availabilityZones: trnAzs,
      },
      instanceType: new ec2.InstanceType(instanceTypeStr),
      machineImage,
      securityGroup: sg,
      role,
      userData,
      // Off by default so a bootstrap edit doesn't silently tear down a healthy
      // trn1 on an unrelated deploy. Opt in (reprovisionOnBootstrapChange) for a
      // deliberate reprovision. See the prop doc for the capacity rationale.
      userDataCausesReplacement: props.reprovisionOnBootstrapChange ?? false,
      keyPair,
      requireImdsv2: true,
      blockDevices: [
        {
          deviceName: '/dev/sda1',
          volume: ec2.BlockDeviceVolume.ebs(200, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
    });
    // NOTE ON REPLACEMENT: this revision moves the Trn1 from the account's
    // default VPC (public subnet) to this stack's own VPC (private isolated
    // subnet). That is a new VPC, new subnet, and new ENI for the instance,
    // so CDK/CloudFormation WILL replace the running instance on first apply
    // of this change, regardless of the pinned logical ID below — a change to
    // `SubnetId`/`VpcId` cannot be applied in place. Plan for capacity
    // headroom (two trn1.2xlarge briefly) exactly as for a deliberate
    // `reprovisionOnBootstrapChange` reprovision, and re-run
    // repoint-agentcore.sh afterwards since the private IP will change.
    //
    // The logical ID is still pinned so that *routine* deploys after this
    // one-time migration (SG tweaks, endpoint additions, non-network changes)
    // continue to update in place rather than replacing the box again.
    (instance.node.defaultChild as ec2.CfnInstance).overrideLogicalId(
      'RewardServer8CE427E6339d8bb32c2d4078',
    );
    Tags.of(instance).add('Name', 'nki-reward-server');
    Tags.of(instance).add('project', 'atx-nki-agent');

    // ── No gateway/NLB/VPC-Link ──────────────────────────────────────────────
    // The AgentCore runtime calls the reward server DIRECTLY at its private
    // IP:5050 (see RewardUrlPrivate below), over the SG-to-SG ingress rule
    // added right after clientSg. The former private-API-Gateway + VPC-Link +
    // NLB fronting stack was removed: its only security contribution was
    // SigV4/IAM auth, which is redundant given the reward server sits in a
    // PRIVATE_ISOLATED subnet (no public IP, no internet route) reachable on
    // 5050 only from the single AgentCore runtime's client SG. Removing it also
    // removes API Gateway's fixed 29-second integration timeout, which capped
    // compile/verify calls that legitimately run for minutes on real hardware.

    new CfnOutput(this, 'InstanceId', { value: instance.instanceId });
    new CfnOutput(this, 'PrivateIp', { value: instance.instancePrivateIp });
    new CfnOutput(this, 'RewardUrlPrivate', {
      description: 'Set this as REWARD_SERVER_URL on the AgentCore runtime (direct, SG-to-SG, no gateway)',
      value: `http://${instance.instancePrivateIp}:5050`,
    });
    new CfnOutput(this, 'BootstrapStatusParam', {
      description: 'SSM String parameter the instance writes bootstrap outcome to (OK / FAILED ...)',
      value: statusParamName,
    });
    new CfnOutput(this, 'AgentCoreClientSgId', {
      description: 'Attach to the AgentCore runtime (VPC mode) for PrivateLink egress (Bedrock/CloudWatch/execute-api)',
      value: clientSg.securityGroupId,
    });
    // Emit the AgentCore ENI subnets (computed once, above) so the runtime is
    // attached to the same subnets the PrivateLink endpoints live in. The
    // runtime still reaches the Trn1 across AZs within the same VPC.
    new CfnOutput(this, 'VpcSubnets', {
      description: 'Subnets for the AgentCore runtime VPC-mode ENIs (private, isolated, AgentCore-supported AZs)',
      value: agentCoreSubnets.map((s) => s.subnetId).join(','),
    });
    new CfnOutput(this, 'CodeArtifactRepositoryArn', {
      description: 'CodeArtifact repository proxying public PyPI for the reward server bootstrap',
      value: `arn:aws:codeartifact:${this.region}:${this.account}:repository/${codeArtifactDomain.domainName}/${codeArtifactRepo.repositoryName}`,
    });
  }
}
