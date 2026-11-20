#!/usr/bin/env bash
# Whole training run, unattended: finish downloads → verify zips → unpack → features → train head.
# Every step skips work already done, so it is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
D=datasets
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for running downloads to finish"
while pgrep -f "download\.sh|curl .*datasets/(slovo|bukva)" >/dev/null; do sleep 60; done
log "downloads: verifying sizes / resuming anything incomplete"
$D/download.sh all

for z in $D/slovo/slovo_full360.zip $D/bukva/bukva.zip; do
  if [ ! -f "$z.ok" ]; then
    log "testing $z"
    unzip -tq "$z" >/dev/null && touch "$z.ok" || { log "BROKEN: $z — delete it and re-run"; exit 1; }
  fi
done

if [ ! -f $D/slovo/.unpacked ]; then
  log "unpacking Slovo"
  unzip -oq $D/slovo/slovo_full360.zip -d $D/slovo/ && touch $D/slovo/.unpacked
fi
if [ ! -f $D/bukva/.unpacked ]; then
  log "unpacking Bukva (trimmed clips + annotations only)"
  unzip -oq $D/bukva/bukva.zip "*trimmed/*" "*annotations*" -d $D/bukva/ && touch $D/bukva/.unpacked
fi

log "features: Slovo"
uv run python scripts/extract_features.py slovo
log "features: Bukva"
uv run python scripts/extract_features.py bukva
log "training the head"
uv run python scripts/train_head.py
log "all done"
