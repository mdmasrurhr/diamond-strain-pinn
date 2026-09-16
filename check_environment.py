"""Record what is needed to reproduce a set of runs.

Writes the interpreter, package versions, hardware, dataset checksum and seed
policy to a manifest file.

Run it before a long study and again afterwards. If the two manifests differ,
the runs span a change of environment and are not a single experiment.

    python check_environment.py
    python check_environment.py --compare results/environment/env_manifest.json

Writes results/environment/env_manifest.json
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import config as C

# Packages whose version can change a number. numpy and torch obviously; sklearn
# because the split and the scalers come from it; pandas because it reads the CSV.
TRACKED = ["numpy", "torch", "pandas", "scikit-learn", "scipy"]


def _sha256(path, cap=None):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def _gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        return [l.strip() for l in out.strip().splitlines()]
    except Exception:
        return []


def manifest():
    m = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict((p, _version(p)) for p in TRACKED),
        "gpus": _gpus(),
        "dataset": {
            "path": os.path.relpath(C.DATA_CSV, C.ROOT),
            "sha256": _sha256(C.DATA_CSV) if os.path.exists(C.DATA_CSV) else None,
            "bytes": os.path.getsize(C.DATA_CSV) if os.path.exists(C.DATA_CSV) else None,
        },
        # The seed policy, not just the seeds: what each one controls matters
        # more than its value.
        "seeds": {
            "list": C.SEEDS,
            "controls": "split partition, weight initialisation, and the labelled "
                        "subset draw (via LABEL_RNG_OFFSET, so the subset does not "
                        "move when the split does)",
            "label_rng_offset": C.LABEL_RNG_OFFSET,
        },
        "thread_policy": {
            "OMP_NUM_THREADS": C.OMP_THREADS,
            "why": "thread count changes floating-point reduction order; the Shi "
                   "baseline batches on the CPU and shifts by ~0.5% if this is "
                   "left to the host",
        },
        "tuned_config": C.TUNED,
    }
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", help="path to an earlier manifest")
    args = ap.parse_args()

    m = manifest()
    print("ENVIRONMENT")
    print("  python    %s  (%s)" % (m["python"], m["python_executable"]))
    print("  platform  %s" % m["platform"])
    for p, v in m["packages"].items():
        print("  %-14s %s" % (p, v or "NOT INSTALLED"))
    for g in m["gpus"]:
        print("  gpu       %s" % g)
    print("  dataset   %s" % m["dataset"]["path"])
    print("            sha256 %s" % (m["dataset"]["sha256"] or "MISSING")[:32])
    print("  seeds     %d (%d..%d)" % (len(C.SEEDS), C.SEEDS[0], C.SEEDS[-1]))

    out = C.results_dir("environment")
    path = os.path.join(out, "env_manifest.json")

    if args.compare and os.path.exists(args.compare):
        old = json.load(open(args.compare))
        diffs = []
        for k in ("python", "platform", "packages", "gpus"):
            if old.get(k) != m.get(k):
                diffs.append((k, old.get(k), m.get(k)))
        if old.get("dataset", {}).get("sha256") != m["dataset"]["sha256"]:
            diffs.append(("dataset sha256",
                          old.get("dataset", {}).get("sha256"),
                          m["dataset"]["sha256"]))
        print("\nCOMPARISON with %s" % args.compare)
        if not diffs:
            print("  identical -- results from before and after are one experiment")
        else:
            for k, a, b in diffs:
                print("  CHANGED %s\n    was: %s\n    now: %s" % (k, a, b))
            print("\n  Results produced either side of this change are not directly")
            print("  comparable. Re-run the affected stages or report them separately.")

    json.dump(m, open(path, "w"), indent=2)
    print("\nwrote %s" % os.path.relpath(path, C.ROOT))


if __name__ == "__main__":
    main()
