#!/bin/bash
#SBATCH -p shared
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x_%A_%a.out
# One line of a task list = one run on one core. Submitted by submit.sh, which sets TASKS (the list of unfinished runs),
# TASK_OFFSET, DONE_FILE, the job name (= task list) and the time and memory limit. Besides the run's own outputs it
# writes into the run folder: command.txt, hardware.txt (node, CPU model, cores, start/end, exit code) and resources.txt
# (wall time, CPU time, peak memory, via /usr/bin/time -v).
set -u
cd "$SLURM_SUBMIT_DIR"
ENV_DIR=${DSPINN_ENV:-$HOME/miniforge3/envs/dspinn}
export PATH="$ENV_DIR/bin:$PATH"
# One thread per run: full-batch training of these small networks is fastest this way, and a fixed thread count keeps
# the floating-point summation order, and so the results, the same on every node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

LINE=$(( ${TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
TASK=$(sed -n "${LINE}p" "$TASKS")
if [ -z "$TASK" ]; then echo "no line $LINE in $TASKS"; exit 1; fi
OUT=$(printf '%s\n' "$TASK" | cut -f1)
CMD=$(printf '%s\n' "$TASK" | cut -f2-)
if [ -f "$OUT/${DONE_FILE:-metrics_summary.csv}" ]; then echo "already done: $OUT"; exit 0; fi
if [ ! -x "$ENV_DIR/bin/python" ]; then echo "no Python environment at $ENV_DIR (run ./setup_env.sh)"; exit 2; fi

mkdir -p "$OUT"
echo "$CMD" > "$OUT/command.txt"
{
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}"
  echo "job_name=${SLURM_JOB_NAME:-?}"
  echo "partition=${SLURM_JOB_PARTITION:-?}"
  echo "cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"
  echo "cores_on_node=$(nproc --all)"
  echo "cores_allocated=$(nproc)"
  echo "omp_threads=$OMP_NUM_THREADS"
  echo "python=$(command -v python)"
  echo "start=$(date -Is)"
} > "$OUT/hardware.txt"
echo "task $LINE of $TASKS -> $OUT on $(hostname)"

if [ -x /usr/bin/time ]; then
  /usr/bin/time -v -o "$OUT/resources.txt" bash -c "$CMD"; rc=$?
else
  s=$(date +%s); bash -c "$CMD"; rc=$?
  echo "Elapsed (wall clock) seconds: $(( $(date +%s) - s ))" > "$OUT/resources.txt"
fi
{ echo "end=$(date -Is)"; echo "exit_code=$rc"; } >> "$OUT/hardware.txt"
exit $rc
