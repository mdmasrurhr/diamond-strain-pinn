"""Run many training jobs across the available GPUs."""

import os
import subprocess
import sys
import time

import config as C


def train_cmd(model, seed, dft_pct, results_dir, extra=None, env_prefix=""):
    """Build one training command.

    `model` is sa, rba or mlp for train_pinn.py, or shi for train_baseline.py.
    Settings come from config.py. Flags in `extra` are appended last, so a
    study varying one of them overrides the default, because argparse keeps
    the final occurrence of a flag.
    """
    script = "train_baseline.py" if model == "shi" else "train_pinn.py"
    cmd = ('%s%s %s --data_csv %s --results_dir %s --seed %d --dft_pct %d'
           % (env_prefix, C.PYTHON, os.path.join(C.ROOT, script),
              C.DATA_CSV, results_dir, seed, dft_pct))
    cmd += " --epochs %d" % C.TUNED["epochs"]
    if model != "shi":
        cmd += " --model %s" % model
        cmd += " --depth %d --width %d" % (C.TUNED["depth"], C.TUNED["width"])
        for k, v in C.LOSS_WEIGHTS.items():
            cmd += " --w_%s %g" % (k, v)
    if extra:
        cmd += " " + " ".join(extra)
    return cmd


def run_jobs(cmds, label, per_gpu=None, gpus=None, log_dir=None):
    """Run `cmds` across the GPUs, at most per_gpu at a time on each.

    Skips any job whose results_dir already holds a metrics_summary.csv, so an
    interrupted study can simply be re-run and will pick up where it stopped.
    """
    gpus = gpus or C.GPUS
    per_gpu = per_gpu or C.RUNS_PER_GPU
    log_dir = log_dir or C.results_dir("logs", label)

    todo = []
    for c in cmds:
        rd = c.split("--results_dir ")[1].split(" ")[0]
        if os.path.exists(os.path.join(rd, "metrics_summary.csv")):
            continue
        todo.append(c)

    done_already = len(cmds) - len(todo)
    print("[%s] %d jobs, %d already done, %d to run, %d at a time"
          % (label, len(cmds), done_already, len(todo), len(gpus) * per_gpu))
    if not todo:
        return

    running = []          # (Popen, gpu, index, logfile handle)
    started = finished = 0
    t0 = time.time()

    # One entry per concurrent slot, holding the GPU that slot belongs to. A
    # finished job hands its slot back, so replacements go to the GPU that just
    # freed up rather than always to the last one in the list.
    free_gpus = [g for g in gpus for _ in range(per_gpu)]

    while finished < len(todo):
        # launch while there is a free slot
        while started < len(todo) and free_gpus:
            gpu = free_gpus.pop(0)
            lf = open(os.path.join(log_dir, "job_%04d.log" % started), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS=str(C.OMP_THREADS),
                       MKL_NUM_THREADS=str(C.OMP_THREADS))
            p = subprocess.Popen(todo[started], shell=True, env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
            running.append((p, gpu, started, lf))
            started += 1

        time.sleep(2)
        for item in list(running):
            p, gpu, i, lf = item
            if p.poll() is not None:
                lf.close()
                running.remove(item)
                free_gpus.append(gpu)          # hand the slot back
                finished += 1
                if p.returncode != 0:
                    print("  ! job %d failed (exit %d) -- see %s/job_%04d.log"
                          % (i, p.returncode, log_dir, i))
                if finished % 25 == 0 or finished == len(todo):
                    el = time.time() - t0
                    rate = finished / el if el else 0
                    left = (len(todo) - finished) / rate if rate else 0
                    print("  %d/%d done (%.0f min elapsed, ~%.0f min left)"
                          % (finished, len(todo), el / 60.0, left / 60.0))
                sys.stdout.flush()

    print("[%s] complete in %.1f min" % (label, (time.time() - t0) / 60.0))
