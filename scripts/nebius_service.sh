#!/usr/bin/env bash
# A systemd unit that restarts a runner at every boot, so a preempted VM picks
# its run back up by itself (every runner skips finished steps and resumes a
# cut-off training from its newest checkpoint). On 2026-10-04 a VM restarted
# after a preemption and sat idle for 3 h 40 min because nothing relaunched the
# run; this is the fix. Run it ON THE VM:
#
#   scripts/nebius_service.sh install nebius_real.py            # unit ctag-run, starts now and at every boot
#   scripts/nebius_service.sh install nebius_real.py CTAG_SYNTH=0 CTAG_REAL_REPEAT=6   # with knobs
#   scripts/nebius_service.sh status                            # is it running, last log lines
#   scripts/nebius_service.sh disable                           # when the run is finished (before stopping the VM)
#
# Log: ~/mtech-thesis-project/logs/<runner>.log (appended across restarts).
set -euo pipefail
DIR="$HOME/mtech-thesis-project"
UNIT=ctag-run
case "${1:-}" in
  install)
    runner=${2:?runner, e.g. nebius_real.py}; shift 2
    [ -f "$DIR/$runner" ] || { echo "no $DIR/$runner" >&2; exit 1; }
    envs=""; for kv in "$@"; do envs+="Environment=$kv"$'\n'; done
    log="$DIR/logs/${runner%.py}.log"
    sudo tee /etc/systemd/system/$UNIT.service >/dev/null <<UNITEOF
[Unit]
Description=ctag runner $runner (resumes after a preemption)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$DIR
Environment=HF_HOME=$DIR/hf_cache
Environment=PATH=$DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin
${envs}ExecStart=/bin/bash -c 'mkdir -p logs && exec python $runner >> $log 2>&1'
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
UNITEOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now $UNIT.service
    sleep 2; systemctl --no-pager status $UNIT.service | head -5
    echo "log: $log"
    ;;
  status)
    systemctl --no-pager status $UNIT.service | head -5 || true
    log=$(grep -o 'exec python [^ ]*' /etc/systemd/system/$UNIT.service 2>/dev/null | awk '{print $3}')
    [ -n "${log:-}" ] && tail -n 5 "$DIR/logs/${log%.py}.log" 2>/dev/null || true
    ;;
  disable)
    sudo systemctl disable --now $UNIT.service 2>/dev/null || true
    echo "$UNIT disabled; the VM can be stopped"
    ;;
  *) echo "usage: $0 install <runner.py> [KEY=VALUE ...] | status | disable" >&2; exit 2;;
esac
