#!/usr/bin/env bash
# Launch N AirSim instances on separate RPC ports and Xvfb displays.
#
# Usage:
#   scripts/launch_airsim_fleet.sh [N_ENVS]
#
# Each instance gets:
#   - Xvfb display :99+i
#   - RPC port 41451 + i*10
#   - Its own settings.json with the correct ApiServerPort
#
# The script waits for all instances to open their RPC ports before exiting.
set -euo pipefail

N_ENVS="${1:-4}"
BASE_PORT=41451
PORT_STEP=10
BASE_DISPLAY=99
SETTINGS_TEMPLATE="${AIRSIM_SETTINGS:-/opt/airsim/settings.json}"

echo "launch_fleet: starting $N_ENVS AirSim instances"

# Find the launcher script
AIRSIM_SH="$(find /opt/airsim -maxdepth 4 -type f \
    \( -name 'Blocks.sh' -o -name 'AirSimNH.sh' -o -name '*.sh' \) \
    -path '*/Linux*/*' 2>/dev/null | head -1)"
if [ -z "$AIRSIM_SH" ]; then
  AIRSIM_SH="$(find /opt/airsim -maxdepth 5 -type f -name '*.sh' | head -1)"
fi
[ -n "$AIRSIM_SH" ] || { echo "launch_fleet: no AirSim launcher found" >&2; exit 1; }
chmod +x "$AIRSIM_SH"

# Kill any existing AirSim/Xvfb processes from a previous run
pkill -f '/opt/airsim' 2>/dev/null || true
pkill -f 'Xvfb' 2>/dev/null || true
sleep 2
pkill -9 -f '/opt/airsim' 2>/dev/null || true
sleep 1

# Create airsim user if it doesn't exist
id -u airsim &>/dev/null || useradd -m -s /bin/bash airsim

for i in $(seq 0 $((N_ENVS - 1))); do
  PORT=$((BASE_PORT + i * PORT_STEP))
  DISPLAY_NUM=$((BASE_DISPLAY + i))

  # Generate per-instance settings.json with the correct port
  INST_SETTINGS="/opt/airsim/settings_${i}.json"
  if command -v python3 >/dev/null; then
    python3 -c "
import json, sys
with open('$SETTINGS_TEMPLATE') as f:
    s = json.load(f)
s['ApiServerPort'] = $PORT
with open('$INST_SETTINGS', 'w') as f:
    json.dump(s, f, indent=2)
"
  else
    # Fallback: sed-based injection (less robust but works)
    cp "$SETTINGS_TEMPLATE" "$INST_SETTINGS"
    if grep -q '"ApiServerPort"' "$INST_SETTINGS"; then
      sed -i "s/\"ApiServerPort\":.*/\"ApiServerPort\": $PORT,/" "$INST_SETTINGS"
    else
      sed -i "s/{/{\"ApiServerPort\": $PORT,/" "$INST_SETTINGS"
    fi
  fi
  chown airsim:airsim "$INST_SETTINGS"

  # Start Xvfb for this instance
  Xvfb :"$DISPLAY_NUM" -screen 0 768x480x24 -ac +extension GLX +render -noreset \
    > "/tmp/xvfb_${i}.log" 2>&1 &

  echo "launch_fleet: instance $i — display :$DISPLAY_NUM, port $PORT"
done

# Wait for all Xvfb displays
for i in $(seq 0 $((N_ENVS - 1))); do
  DISPLAY_NUM=$((BASE_DISPLAY + i))
  for attempt in $(seq 1 30); do
    if xdpyinfo -display :"$DISPLAY_NUM" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  xdpyinfo -display :"$DISPLAY_NUM" >/dev/null 2>&1 || {
    echo "launch_fleet: Xvfb :$DISPLAY_NUM never came up" >&2
    exit 1
  }
done

# Launch AirSim instances
for i in $(seq 0 $((N_ENVS - 1))); do
  PORT=$((BASE_PORT + i * PORT_STEP))
  DISPLAY_NUM=$((BASE_DISPLAY + i))
  INST_SETTINGS="/opt/airsim/settings_${i}.json"

  chown -R airsim:airsim /opt/airsim "$HOME/Documents/AirSim" 2>/dev/null || true

  DISPLAY=:"$DISPLAY_NUM" nohup runuser -u airsim -- "$AIRSIM_SH" \
      -settings="$INST_SETTINGS" \
      -opengl4 \
      -windowed -ResX=768 -ResY=480 \
      -nosound -unattended -nosplash \
      > "/tmp/airsim_${i}.log" 2>&1 &

  echo "launch_fleet: AirSim $i started (PID $!, log /tmp/airsim_${i}.log)"
done

# Wait for all RPC ports
echo "launch_fleet: waiting for RPC ports..."
for i in $(seq 0 $((N_ENVS - 1))); do
  PORT=$((BASE_PORT + i * PORT_STEP))
  for attempt in $(seq 1 120); do
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
      echo "launch_fleet: instance $i ready on port $PORT (${attempt}s)"
      break
    fi
    sleep 1
  done
  nc -z 127.0.0.1 "$PORT" 2>/dev/null || {
    echo "launch_fleet: instance $i never opened port $PORT" >&2
    tail -20 "/tmp/airsim_${i}.log"
    exit 1
  }
done

echo "launch_fleet: all $N_ENVS instances ready"
# Write port list for train.py to read
PORT_LIST=""
for i in $(seq 0 $((N_ENVS - 1))); do
  PORT=$((BASE_PORT + i * PORT_STEP))
  PORT_LIST="${PORT_LIST:+$PORT_LIST,}$PORT"
done
echo "$PORT_LIST" > /tmp/airsim_ports.txt
echo "launch_fleet: port list written to /tmp/airsim_ports.txt: $PORT_LIST"
