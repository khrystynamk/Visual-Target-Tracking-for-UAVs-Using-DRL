#!/usr/bin/env bash
# Gracefully stop training on a vast.ai instance.
# Sends SIGTERM to train.py, which triggers the finally block
# (saves final model + replay buffer, uploads to R2).
#
# Usage (on the instance):
#   bash scripts/r2_stop.sh
#
# Or from your Mac via SSH:
#   ssh -p PORT root@HOST 'cd /workspace/vtt && bash scripts/r2_stop.sh'
set -euo pipefail

TRAIN_PID="$(pgrep -of 'python scripts/train.py' || true)"

if [ -z "$TRAIN_PID" ]; then
  echo "r2_stop: no training process found"
  exit 0
fi

echo "r2_stop: sending SIGTERM to train.py (PID $TRAIN_PID)"
kill -TERM "$TRAIN_PID"

# Wait for process to finish (model save + R2 upload)
echo "r2_stop: waiting for graceful shutdown..."
TIMEOUT=300  # 5 minutes max for replay buffer upload
for i in $(seq 1 $TIMEOUT); do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "r2_stop: training stopped cleanly after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "r2_stop: WARNING — process did not exit within ${TIMEOUT}s, sending SIGKILL" >&2
echo "r2_stop: WARNING — final R2 upload did NOT complete; latest checkpoint may be stale" >&2
kill -9 "$TRAIN_PID" 2>/dev/null || true
exit 1
