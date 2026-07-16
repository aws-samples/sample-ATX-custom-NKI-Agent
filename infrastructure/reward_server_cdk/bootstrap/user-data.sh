#!/usr/bin/env bash
# Cloud-init user-data for the Trn1 reward server.
#
# Placeholders (__UPPER__) are substituted by the CDK stack at synth time.
# Runs once at first boot on a stock Neuron Deep Learning AMI:
#   1. download the reward server source bundle from the CDK S3 asset
#   2. authenticate pip against the CodeArtifact PyPI proxy and install the
#      reward server into the Neuron venv
#   3. run the server under systemd (survives reboot, single worker)
#
# Source is pulled from S3 (not git) so a fresh instance needs no repo
# credentials — works with private repos the box can't clone.
#
# The instance has no public IP and no route to the internet (see
# reward-server-stack.ts): every AWS call below (s3, ssm, codeartifact) goes
# over a PrivateLink interface/gateway endpoint in this VPC, and pip resolves
# packages through the CodeArtifact repository (which proxies public PyPI on
# CodeArtifact's own egress, not the instance's) instead of pypi.org directly.
set -euxo pipefail

ASSET_BUCKET="__ASSET_BUCKET__"
ASSET_KEY="__ASSET_KEY__"
STATUS_PARAM="__STATUS_PARAM__"
REGION="__REGION__"
CODEARTIFACT_DOMAIN="__CODEARTIFACT_DOMAIN__"
CODEARTIFACT_DOMAIN_OWNER="__CODEARTIFACT_DOMAIN_OWNER__"
CODEARTIFACT_REPO="__CODEARTIFACT_REPO__"
APP_DIR="/opt/nki-agent"
LOG=/var/log/nki-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

# Fail loudly: if any step aborts, record the failing line in SSM so the
# re-point script and operators get a clear signal instead of a silent
# failure.
on_err() {
  local line=$1
  aws ssm put-parameter --region "$REGION" --name "$STATUS_PARAM" \
    --type String --overwrite \
    --value "FAILED at line ${line}; see /var/log/nki-bootstrap.log" || true
}
trap 'on_err $LINENO' ERR

# The Neuron DLAMI ships an activated venv with torch-neuronx + neuronx-cc.
# Prefer it so compile/verify/profile use the SDK toolchain already on the box.
# The reward server spawns subprocesses via sys.executable, so the SERVICE must
# run under this venv's Python — otherwise handlers can't import numpy/torch/
# neuronxcc. We resolve the venv here and use it for both install and systemd.
VENV=""
for cand in /opt/aws_neuronx_venv_pytorch_* /opt/aws_neuron_venv_pytorch_*; do
  if [ -x "$cand/bin/python" ]; then VENV="$cand"; break; fi
done
if [ -n "$VENV" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  PYBIN="$VENV/bin/python"
else
  PYBIN="$(command -v python3)"
fi
echo "using python: $PYBIN (venv=${VENV:-none})"
"$PYBIN" -c "import numpy, torch" >/dev/null 2>&1 \
  && echo "numpy+torch import OK" \
  || echo "WARN: numpy/torch not importable in $PYBIN"

# 1. Fetch the reward_server source bundle from the CDK S3 asset.
#    The asset zips the reward_server/ directory, so it unpacks to $APP_DIR/reward_server.
rm -rf "$APP_DIR" && mkdir -p "$APP_DIR/reward_server"
aws s3 cp "s3://${ASSET_BUCKET}/${ASSET_KEY}" /tmp/reward_server.zip --region "$REGION"
unzip -o /tmp/reward_server.zip -d "$APP_DIR/reward_server"

# 2. Install reward-server deps (flask + gunicorn; torch/neuron already present)
#    through the CodeArtifact PyPI proxy — the instance has no route to
#    pypi.org directly, so `aws codeartifact login` configures pip's index-url
#    to the CodeArtifact repository endpoint (over the PrivateLink endpoints
#    provisioned by the stack) instead. --ignore-installed avoids aborting on
#    distutils-managed system packages (e.g. blinker 1.4) that pip cannot
#    cleanly uninstall.
aws codeartifact login --region "$REGION" --tool pip \
  --domain "$CODEARTIFACT_DOMAIN" --domain-owner "$CODEARTIFACT_DOMAIN_OWNER" \
  --repository "$CODEARTIFACT_REPO"
"$PYBIN" -m pip install --upgrade pip
"$PYBIN" -m pip install --ignore-installed \
  -r "$APP_DIR/reward_server/requirements.txt"

# 3. systemd unit — 4 workers (see ExecStart below for why single-worker was
#    dropped; the persisted-NEFF write path is request-scoped so concurrent
#    workers can't collide), bound to 0.0.0.0 but the security group locks
#    5050 to the VPC.
cat >/etc/systemd/system/nki-reward.service <<UNIT
[Unit]
Description=ATX NKI reward server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
Environment=REWARD_HOST=0.0.0.0
Environment=REWARD_PORT=5050
# Network is the trust boundary (private-isolated subnet + SG-to-SG on 5050,
# no API Gateway in front). App-layer auth defaults off; set true only behind
# an identity-injecting proxy. See infrastructure/reward_server/auth.py.
Environment=REWARD_REQUIRE_AUTH=false
# Put the Neuron venv first on PATH so gunicorn AND the compile/verify/profile
# subprocesses (spawned via sys.executable) inherit numpy / torch / neuronxcc.
Environment=PATH=${VENV:+$VENV/bin:}/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin
# 4 workers so concurrent compile/verify/profile calls don't queue behind one
# another (a single NKI compile holds a worker for seconds; -w 1 serialized every
# request and made the second concurrent caller time out). This is safe against
# the NEFF-cache collision that motivated -w 1 originally: compile.py persists
# each successful compile's NEFF under a per-request UUID-suffixed filename
# (reward_server/compile.py), so two workers compiling the same kernel_name
# concurrently can no longer overwrite each other's output. --timeout 600
# raises gunicorn's own worker timeout well above the default 30s: an NKI
# compile/verify can legitimately run longer than 30s, and the default would kill
# the worker mid-compile (SIGKILL -> the caller sees a dropped connection / 5xx)
# even though the compile would have succeeded. NOTE: API Gateway's REST
# integration used to cap a single call at 29s; that gateway has been removed
# (SG-to-SG direct calls, see reward-server-stack.ts), so this constraint no
# longer applies.
ExecStart=${PYBIN} -m gunicorn -w 4 --timeout 600 --graceful-timeout 600 -b 0.0.0.0:5050 reward_server.server:app
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now nki-reward.service

# Wait for the server to come up, then require /health/deep to pass — this
# imports numpy/torch/neuronxcc in the SERVICE interpreter, so a misconfigured
# venv fails provisioning loudly instead of silently serving failing compiles.
healthy=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:5050/health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 5
done
if [ "$healthy" != "1" ]; then
  aws ssm put-parameter --region "$REGION" --name "$STATUS_PARAM" \
    --type String --overwrite --value "FAILED: server did not become healthy" || true
  exit 1
fi

# The deep check imports the heavy Neuron stack (torch + neuronxcc) in the
# single worker on first hit — that can take a minute cold. Retry with a
# generous per-attempt timeout instead of giving up after one slow request.
deep_ok=0
DEEP=""
for _ in $(seq 1 12); do
  DEEP="$(curl -fsS --max-time 90 http://127.0.0.1:5050/health/deep 2>/dev/null || true)"
  if echo "$DEEP" | grep -qE '"status":\s*"ok"'; then deep_ok=1; break; fi
  sleep 10
done
echo "health/deep: $DEEP"
if [ "$deep_ok" != "1" ]; then
  aws ssm put-parameter --region "$REGION" --name "$STATUS_PARAM" \
    --type String --overwrite --value "FAILED: toolchain import check (see /health/deep)" || true
  exit 1
fi
echo "reward server healthy (toolchain OK)"
aws ssm put-parameter --region "$REGION" --name "$STATUS_PARAM" \
  --type String --overwrite --value "OK"
