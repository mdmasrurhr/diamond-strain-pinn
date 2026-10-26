#!/bin/bash
# Usage: ./submit.sh tasks/<list>.tsv [max_running]       (run from this folder on the login node of a SLURM cluster)
# Submits the runs of one task list that are not finished yet, as SLURM job arrays of single-core jobs (split into
# arrays of <= the cluster's MaxArraySize, at most 1000). At most max_running (default 400) run at once. The paper's
# runs used the shared partition of LEAP2 (Texas State University); set PARTITION for another cluster.
# Safe to re-run after time-outs or failures: finished runs are skipped. Refuses while jobs of the same list are still
# queued or running, because a second copy of a run would write into the same folder.
# Optional overrides: TIME=hh:mm:ss MEM=2G PARTITION=shared ACCOUNT=<account> CHUNK=1000
# DEPEND=afterany:<jobid>:<jobid> makes the list wait until those jobs have ended (job ids are printed at the end and
# saved in tasks/todo/<list>.jobids), e.g. DEPEND=afterany:$(cat tasks/todo/main.jobids) ./submit.sh tasks/ablation.tsv
set -eu
cd "$(dirname "$0")"
TASKS=${1:?usage: ./submit.sh tasks/<list>.tsv [max_running]}
MAXC=${2:-400}
[ -f "$TASKS" ] || { echo "no such task list: $TASKS"; exit 1; }
NAME=$(basename "$TASKS" .tsv)

# Time limit per run: about 3x the longest run measured on the workstation (constants as outputs ~50 min, the others <= 30 min).
case "$NAME" in
  smoke)  T=00:30:00; M=2G; DF=metrics_summary.csv ;;
  *)      if grep -q -- "--structured_head" "$TASKS"; then T=04:00:00; else T=02:00:00; fi; M=2G; DF=metrics_summary.csv ;;
esac
TIME=${TIME:-$T}; MEM=${MEM:-$M}; PARTITION=${PARTITION:-shared}; DONE_FILE=${DONE_FILE:-$DF}

if [ -n "$(squeue -u "$USER" -h -n "$NAME" -o %i)" ]; then
  echo "jobs of $NAME are still queued or running: wait (./status.sh $TASKS) or cancel them first (scancel -n $NAME)"; exit 1
fi
mkdir -p logs tasks/todo
TODO=tasks/todo/${NAME}_$(date +%Y%m%d_%H%M%S).tsv
while IFS= read -r line; do
  out=${line%%$'\t'*}
  [ -f "$out/$DONE_FILE" ] || printf '%s\n' "$line"
done < "$TASKS" > "$TODO"
N=$(wc -l < "$TODO"); TOT=$(wc -l < "$TASKS")
echo "$NAME: $((TOT - N)) of $TOT runs already finished; submitting $N (partition $PARTITION, time limit $TIME, memory $MEM, at most $MAXC at once)"
if [ "$N" -eq 0 ]; then rm -f "$TODO"; exit 0; fi

MAXARR=$(scontrol show config 2>/dev/null | awk '/^MaxArraySize/ {print $3}')
CH=${CHUNK:-$(( ${MAXARR:-1001} - 1 ))}; [ "$CH" -gt 1000 ] && CH=1000
ACC=""; [ -n "${ACCOUNT:-}" ] && ACC="--account=$ACCOUNT"
DEP=""; [ -n "${DEPEND:-}" ] && DEP="--dependency=$DEPEND"
IDS=""
for ((off = 0; off < N; off += CH)); do
  n=$(( N - off < CH ? N - off : CH ))
  PER=$(( (MAXC * n + N - 1) / N ))                                # the limit is shared between the arrays by size
  id=$(sbatch --parsable $ACC $DEP -J "$NAME" -p "$PARTITION" -t "$TIME" --mem="$MEM" --array="1-${n}%${PER}" \
         --export=ALL,TASKS="$TODO",TASK_OFFSET=$off,DONE_FILE="$DONE_FILE" run_task.sh)
  IDS="${IDS:+$IDS:}${id%%;*}"
done
echo "$IDS" > "tasks/todo/$NAME.jobids"
echo "submitted job id(s) $IDS${DEP:+ ($DEP)}; follow with: ./status.sh $TASKS"
