# Trn1 reward server

Long-lived Flask service that runs on a Trn1 instance (Trn1.2xlarge or larger). It is the only component in the system that touches real Neuron hardware, and it is the single source of truth for the agent's compile / verify / profile signals.

## Endpoints

```
POST /reward         { "op": "compile" | "verify" | "profile" | "skill", ... }
POST /reward/batch   { "items": [ ... ] }
GET  /health
```

## Deploy

On a Trn1.2xlarge with the Neuron SDK installed:

```bash
pip install -r requirements.txt
gunicorn -w 4 -b 0.0.0.0:5050 reward_server.server:app
```

Run as a systemd unit for production. Lock the security group down to the AgentCore VPC and (optionally) the AWS Batch container security group for the ATX Custom remote-execution path.

## Auth

Default auth is "private subnet only". For internet-facing deployments, terminate TLS at an ALB and require SigV4 signatures from callers (the MCP server signs outbound requests with the standard AWS credential chain).

## Production bodies

The four worker modules ship as stubs in this repo. For a production deployment, drop in the implementations from `kernel-forge/src/env/`:

- `compile.py`     ← `kernel-forge/src/env/compiler_wrapper.py`
- `verify.py`      ← `kernel-forge/src/env/verifier.py`
- `profile.py`     ← `kernel-forge/src/env/profiler_wrapper.py`
- `skill.py`       ← back with OpenSearch or a vector store
