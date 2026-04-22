#!/usr/bin/env bash
# Kill and relaunch AirSim on the vast.ai instance.
# Called by TrackingEnv._reconnect() when AirSim becomes unresponsive.
# Also safe to run manually from SSH.
set -euo pipefail

SETTINGS_PATH="${AIRSIM_SETTINGS:-/opt/airsim/settings.json}"

# --- Kill any existing AirSim processes --------------------------------------
echo "restart_airsim: killing existing AirSim processes..."
pkill -f '/opt/airsim.*\.sh' 2>/dev/null || true
pkill -f 'CrashReportClient' 2>/dev/null || true
# Give UE a moment to die
sleep 2
# Force-kill if still alive
pkill -9 -f '/opt/airsim' 2>/dev/null || true
sleep 1

# --- Find the launcher script ------------------------------------------------
AIRSIM_SH="$(find /opt/airsim -maxdepth 4 -type f \
    \( -name 'Blocks.sh' -o -name 'AirSimNH.sh' -o -name '*.sh' \) \
    -path '*/Linux*/*' 2>/dev/null | head -1)"
if [ -z "$AIRSIM_SH" ]; then
  AIRSIM_SH="$(find /opt/airsim -maxdepth 5 -type f -name '*.sh' | head -1)"
fi
[ -n "$AIRSIM_SH" ] || { echo "restart_airsim: no launcher found" >&2; exit 1; }
chmod +x "$AIRSIM_SH"

# --- Relaunch ----------------------------------------------------------------
echo "restart_airsim: launching $AIRSIM_SH"
DISPLAY=:99 nohup runuser -u airsim -- "$AIRSIM_SH" \
    -settings="$SETTINGS_PATH" \
    -opengl4 \
    -windowed -ResX=768 -ResY=480 \
    -nosound -unattended -nosplash \
    > /tmp/airsim.log 2>&1 &

# --- Wait for RPC port -------------------------------------------------------
for i in $(seq 1 120); do
  if nc -z 127.0.0.1 41451 2>/dev/null; then
    echo "restart_airsim: RPC ready after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "restart_airsim: AirSim did not open port 41451 within 120s" >&2
exit 1
