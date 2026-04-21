#!/usr/bin/env bash
# Provisions a vast.ai instance for SAC training against an AirSim Linux binary
# pulled from Cloudflare R2, then starts training.
#
# Required env vars (set in vast.ai template):
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY
#   R2_ACCOUNT_ID
#   WANDB_API_KEY
#
# Optional env vars:
#   CONFIG       - training config path (default: configs/drl/sac_depth.yaml)
#   UE_PKG_KEY   - R2 key for the AirSim zip (default: ue-packages/Blocks.zip)
#   BUCKET       - R2 bucket (default: vtt-uav-artifacts)
set -euxo pipefail

: "${CONFIG:=configs/drl/sac_depth.yaml}"
: "${UE_PKG_KEY:=ue-packages/Blocks.zip}"
: "${BUCKET:=vtt-uav-artifacts}"

for v in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ACCOUNT_ID WANDB_API_KEY; do
  if [ -z "${!v:-}" ]; then
    echo "bootstrap_vastai: required env var $v is empty" >&2
    exit 2
  fi
done

# --- System deps for UE Linux Shipping binaries ------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  libvulkan1 vulkan-tools mesa-vulkan-drivers \
  libsdl2-2.0-0 libasound2t64 libpulse0 \
  libxrandr2 libxi6 libxinerama1 libxcursor1 \
  unzip curl ca-certificates git netcat-openbsd tmux

# --- AWS CLI v2 -------------------------------------------------------------
if ! command -v aws >/dev/null; then
  curl -L "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp
  /tmp/aws/install
fi

mkdir -p "$HOME/.aws"
cat > "$HOME/.aws/credentials" <<CREDS
[r2]
aws_access_key_id=${R2_ACCESS_KEY_ID}
aws_secret_access_key=${R2_SECRET_ACCESS_KEY}
CREDS
cat > "$HOME/.aws/config" <<CONF
[profile r2]
region=auto
endpoint_url=https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
output=json
CONF

# --- Pull the AirSim binary from R2 -----------------------------------------
mkdir -p /opt/airsim
aws --profile r2 s3 cp "s3://${BUCKET}/${UE_PKG_KEY}" /tmp/airsim.zip
rm -rf /opt/airsim
unzip -q /tmp/airsim.zip -d /opt/airsim

# Find the launcher script wherever it landed (names vary across AirSim
# releases and between stock vs custom packages).
AIRSIM_SH="$(find /opt/airsim -maxdepth 4 -type f \
    \( -name 'Blocks.sh' -o -name 'AirSimNH.sh' -o -name '*.sh' \) \
    -path '*/Linux*/*' 2>/dev/null | head -1)"
if [ -z "$AIRSIM_SH" ]; then
  AIRSIM_SH="$(find /opt/airsim -maxdepth 5 -type f -name '*.sh' | head -1)"
fi
[ -n "$AIRSIM_SH" ] || { echo "bootstrap_vastai: no AirSim launcher found" >&2; ls -R /opt/airsim | head -40; exit 1; }
chmod +x "$AIRSIM_SH"
echo "bootstrap_vastai: AirSim launcher = $AIRSIM_SH"

# --- AirSim settings.json ----------------------------------------------------
mkdir -p "$HOME/Documents/AirSim"
cp configs/airsim/settings.json "$HOME/Documents/AirSim/settings.json"

# --- Launch AirSim headless (UE refuses root) --------------------------------
id -u airsim &>/dev/null || useradd -m -s /bin/bash airsim
chown -R airsim:airsim /opt/airsim "$HOME/Documents/AirSim"
nohup runuser -u airsim -- "$AIRSIM_SH" \
    -RenderOffScreen -nosound -windowed -ResX=1024 -ResY=768 \
    > /tmp/airsim.log 2>&1 &

# --- Wait for RPC port -------------------------------------------------------
for i in $(seq 1 120); do
  if nc -z 127.0.0.1 41451 2>/dev/null; then
    echo "bootstrap_vastai: AirSim RPC ready after ${i}s"
    break
  fi
  sleep 1
done
nc -z 127.0.0.1 41451 || {
  echo "bootstrap_vastai: AirSim never opened port 41451"
  tail -80 /tmp/airsim.log
  exit 1
}

# --- Python deps -------------------------------------------------------------
pip install torch torchvision
pip install msgpack-rpc-python
pip install --no-build-isolation airsim
pip install -r requirements.txt
pip install -e .

# --- W&B + train -------------------------------------------------------------
wandb login "$WANDB_API_KEY"

# Training runs in the foreground so vast.ai's onstart log shows it.
# Use --no-render because the instance is headless.
exec python scripts/train.py --config "$CONFIG" --no-render
