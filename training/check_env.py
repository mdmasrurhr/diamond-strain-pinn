"""Check the environment and the task lists in a few seconds (safe on a cluster login node).

Checks the Python and package versions, the dataset (SHA-256), and every task list in tasks/: each constants file a run
needs exists and holds the 13 constants, each option is accepted by train.py, and no two runs share a results folder.
Prints OK / WARN / FAIL lines and exits with 1 if anything failed.    Usage: python check_env.py
"""

import glob
import hashlib
import importlib
import json
import os
import re
import sys

import config as C

os.chdir(C.HERE)
failed = []


def report(level, msg):
    print("%-5s %s" % (level, msg))
    if level == "FAIL":
        failed.append(msg)


# packages: the versions that produced the paper's results
WANT = {
    "torch": "2.11.0",
    "numpy": "2.4.4",
    "pandas": "3.0.2",
    "sklearn": "1.8.0",
    "scipy": "1.17.1",
    "matplotlib": "3.10.8",
    "psutil": None,
}
report("OK" if sys.version_info[:2] == (3, 12) else "WARN", "Python %s (paper: 3.12)" % sys.version.split()[0])
for name, version in WANT.items():
    try:
        module = importlib.import_module(name)
        have = module.__version__.split("+")[0]
        report(
            "OK" if version in (None, have) else "WARN",
            "%s %s" % (name, module.__version__) + ("" if version in (None, have) else ", expected %s" % version),
        )
    except ImportError as exc:
        report("WARN" if name == "psutil" else "FAIL", "%s cannot be imported: %s" % (name, exc))
try:
    import torch

    x = torch.randn(64, 6, requires_grad=True)
    torch.nn.Linear(6, 1)(x).sum().backward()
    report("OK", "torch forward and backward pass on the CPU")
except Exception as exc:
    report("FAIL", "torch test: %s" % exc)

# dataset
SHA = "44dacb84df6a4924f236f1d42863e9b328957dd8adbc50e453c51d44eae0ac8c"
if not os.path.exists(C.DATA_CSV):
    report("FAIL", "missing " + C.DATA_CSV)
elif hashlib.sha256(open(C.DATA_CSV, "rb").read()).hexdigest() != SHA:
    report("FAIL", "dft_hy_v9.csv differs from the published dataset (SHA-256)")
else:
    report("OK", "dataset dft_hy_v9.csv (SHA-256 matches)")

# task lists
KEYS = {"CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2", "VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"}
OPTIONS = set(re.findall(r'add_argument\("(--\w+)"', open("train.py").read()))
for path in sorted(glob.glob("tasks/*.tsv")):
    problems, folders, constants = set(), set(), {}
    lines = [line.rstrip("\n") for line in open(path) if line.strip()]
    for line in lines:
        out, cmd = line.split("\t", 1)
        if out in folders:
            problems.add("results folder used twice: " + out)
        folders.add(out)
        f = re.search(r"PHYSICS_CONST_JSON=(\S+)", cmd).group(1)
        if f not in constants:
            try:
                constants[f] = KEYS <= set(json.load(open(f)))
            except (OSError, ValueError):
                constants[f] = False
        if not constants[f]:
            problems.add("constants file missing or incomplete: " + f)
        for option in set(re.findall(r"(?<!\S)(--\w+)", cmd)) - OPTIONS:
            problems.add("train.py does not accept " + option)
        data = re.search(r"--data_csv (\S+)", cmd).group(1)
        if not os.path.exists(data):
            problems.add("missing data file " + data)
    for p in sorted(problems)[:10]:
        report("FAIL", "%s: %s" % (path, p))
    if not problems:
        report("OK", "%s: %d runs, all files and options present" % (path, len(lines)))

print("\n" + ("ALL CHECKS PASSED" if not failed else "%d CHECK(S) FAILED" % len(failed)))
sys.exit(1 if failed else 0)
