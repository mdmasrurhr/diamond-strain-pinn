#!/bin/bash
# Usage: ./status.sh tasks/<list>.tsv   — finished, failed, running and waiting runs of one task list.
# "failed" = the run ended with an error; "to resubmit" = not finished and no longer queued (failed or timed out).
cd "$(dirname "$0")"
TASKS=${1:?usage: ./status.sh tasks/<list>.tsv}
NAME=$(basename "$TASKS" .tsv)
DF=metrics_summary.csv
done_n=0; failed=0; tot=0; examples=()
while IFS=$'\t' read -r out cmd; do
  tot=$((tot + 1))
  if [ -f "$out/$DF" ]; then done_n=$((done_n + 1))
  elif [ -f "$out/hardware.txt" ] && grep -q '^exit_code=[1-9]' "$out/hardware.txt"; then
    failed=$((failed + 1)); [ ${#examples[@]} -lt 3 ] && examples+=("$out")
  fi
done < "$TASKS"
running=$(squeue -u "$USER" -h -r -n "$NAME" -t R | wc -l)
waiting=$(squeue -u "$USER" -h -r -n "$NAME" -t PD | wc -l)
left=$(( tot - done_n - running - waiting )); [ $left -lt 0 ] && left=0
echo "$NAME: $done_n / $tot finished | $running running | $waiting waiting | $failed failed | $left to resubmit"
for o in "${examples[@]}"; do
  j=$(sed -n 's/^slurm_job=//p' "$o/hardware.txt")
  echo "  failed: $o   log: logs/${NAME}_${j}.out"
done
[ $left -gt 0 ] && [ $((running + waiting)) -eq 0 ] && echo "  -> resubmit the unfinished runs: ./submit.sh $TASKS"
exit 0
