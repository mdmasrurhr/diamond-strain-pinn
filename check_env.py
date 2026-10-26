#!/usr/bin/env python3
"""Environment and repository check. Takes seconds (safe on a cluster login node).
Checks the Python and package versions, the dataset (SHA-256 of the v9 file), and every task list in tasks/: each
constants file a run needs exists and holds the 13 constants, each flag exists in the script it is passed to, and no two
runs share a results folder. Prints OK / WARN / FAIL lines; exit code 1 if anything failed.
Usage: python check_env.py"""
import glob, hashlib, importlib, json, os, re, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
failed = []


def ok(msg):
    print("OK    " + msg)


def warn(msg):
    print("WARN  " + msg)


def fail(msg):
    print("FAIL  " + msg)
    failed.append(msg)


# ---- Python and packages (the versions that produced every result of the paper)
WANT = {"torch": "2.11.0", "numpy": "2.4.4", "pandas": "3.0.2", "sklearn": "1.8.0", "scipy": "1.17.1",
        "matplotlib": "3.10.8", "psutil": None}
py = sys.version.split()[0]
(ok if sys.version_info[:2] == (3, 12) else warn)("Python %s (paper: 3.12)" % py)
for name, version in WANT.items():
    try:
        mod = importlib.import_module(name)
        have = mod.__version__.split("+")[0]
        if version is None or have == version:
            ok("%s %s" % (name, mod.__version__))
        else:
            warn("%s %s, expected %s" % (name, mod.__version__, version))
    except Exception as e:
        (warn if name == "psutil" else fail)("%s cannot be imported: %s%s" % (name, e, " (optional: only the CPU and memory columns of epoch_log.csv need it)" if name == "psutil" else ""))
try:
    import torch
    x = torch.randn(64, 6, requires_grad=True)
    torch.nn.Linear(6, 1)(x).sum().backward()
    ok("torch CPU forward and backward pass (%d thread(s) here; jobs use 1)" % torch.get_num_threads())
except Exception as e:
    fail("torch test: %s" % e)

# ---- dataset
DATA, SHA = "data/dft_hy_v9.csv", "44dacb84df6a4924f236f1d42863e9b328957dd8adbc50e453c51d44eae0ac8c"
if not os.path.exists(DATA):
    fail("missing " + DATA)
elif hashlib.sha256(open(DATA, "rb").read()).hexdigest() != SHA:
    fail(DATA + " differs from the v9 dataset (SHA-256)")
else:
    ok(DATA + " is the v9 dataset (SHA-256 matches)")

# ---- task lists
KEYS = {"CBM_0", "XI_D", "XI_U", "XI_UP", "XI_Q", "XI_C", "XI_Q2", "VBM_0", "AV", "AV2", "B_BP", "B2_BP", "D_BP"}


def flags_of(script):
    src = open(script).read()
    found = set()
    for call in re.findall(r"add_argument\(([^)]*)", src):
        found.update(re.findall(r"[\"'](--[\w-]+)[\"']", call))
    return found


FLAGS = {s: flags_of(s) for s in ["03_train_pinn.py"]}
seen, const_ok = {}, {}
for path in sorted(glob.glob("tasks/*.tsv")):
    n, problems = 0, set()
    for line in open(path):
        if not line.strip():
            continue
        n += 1
        out, cmd = line.rstrip("\n").split("\t", 1)
        if out in seen and seen[out][1] != cmd:          # the same run listed again (identical command) is fine
            problems.add("results folder %s also used by a different command in %s" % (out, seen[out][0]))
        seen.setdefault(out, (path, cmd))
        m = re.search(r"PHYSICS_CONST_JSON=(\S+)", cmd)
        if m:
            f = m.group(1)
            if f not in const_ok:
                try:
                    const_ok[f] = KEYS <= set(json.load(open(f)))
                except Exception:
                    const_ok[f] = False
            if not const_ok[f]:
                problems.add("constants file missing or incomplete: " + f)
        for script in re.findall(r"python (\S+\.py)", cmd):
            if not os.path.exists(script):
                problems.add("missing script " + script)
            elif script in FLAGS:
                for fl in set(re.findall(r"(?<!\S)(--[\w-]+)", cmd.split(script, 1)[1].split("&&")[0])) - FLAGS[script]:
                    problems.add("%s does not accept %s" % (script, fl))
        d = re.search(r"--data_csv (\S+)", cmd)
        if d and not os.path.exists(d.group(1)):
            problems.add("missing data file " + d.group(1))
    if problems:
        for p in sorted(problems)[:10]:
            fail("%s: %s" % (path, p))
    else:
        ok("%s: %d runs, all files and flags present, no folder shared with a different run" % (path, n))

print("\n%s" % ("ALL CHECKS PASSED" if not failed else "%d CHECK(S) FAILED" % len(failed)))
sys.exit(1 if failed else 0)
