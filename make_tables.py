"""
make_tables.py -- gather every finished run into a few readable tables.

Each training run writes its own metrics_summary.csv into its own folder. This
script walks results/, reads all of them, and produces the aggregated tables the
figures and the paper are written from. Run it whenever a study finishes; it is
cheap and safe to repeat.

    python make_tables.py

Writes into results/tables/
    all_runs.csv        one row per training run, every study, with its study tag
    sweep.csv           accuracy vs labelled fraction, mean +/- s.d. over seeds
    ablations.csv       every ablation variant against its reference
    per_strain.csv      full-data error split by deformation family
"""

import glob
import os

import numpy as np
import pandas as pd

import config as C

# Where each study writes, and how to read a run's identity back out of its path.
STUDIES = {
    "label_sweep":           ("sweep",         ["model", "pct", "seed"]),
    "loss_ablation":         ("ablation",      ["variant", "pct", "seed"]),
    "design_ablation":       ("ablation",      ["variant", "pct", "seed"]),
    "hyperparameter_search": ("hpsearch",      ["trial", "seed"]),
    "matched_prior":         ("label_matched", ["model", "pct", "seed"]),
    "duplicate_ablation":    ("dedup",         ["model", "pct", "seed"]),
}


def collect():
    rows = []
    for study, (kind, fields) in STUDIES.items():
        base = os.path.join(C.RESULTS, study)
        for f in glob.glob(os.path.join(base, "**", "metrics_summary.csv"),
                           recursive=True):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if not len(d):
                continue
            r = d.iloc[-1].to_dict()
            # the folder names between the study dir and the file are the identity
            parts = os.path.relpath(os.path.dirname(f), base).split(os.sep)
            r["study"] = study
            r["kind"] = kind
            for i, name in enumerate(fields):
                r[name] = parts[i] if i < len(parts) else ""
            r["_path"] = f
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    out = C.results_dir("tables")
    d = collect()
    if not len(d):
        print("no finished runs found under results/ -- run 05 first")
        return

    # normalise the two identity columns used for grouping
    if "pct" in d.columns:
        d["pct_num"] = pd.to_numeric(
            d["pct"].astype(str).str.replace("pct", "", regex=False),
            errors="coerce")
    d["mae"] = pd.to_numeric(d["test_mae_meV"], errors="coerce")
    d.to_csv("%s/all_runs.csv" % out, index=False)
    print("collected %d runs across %d studies"
          % (len(d), d["study"].nunique()))

    # ---- the sweep ------------------------------------------------------
    sw = d[d["kind"] == "sweep"]
    if len(sw):
        g = (sw.groupby(["model", "pct_num"])["mae"]
               .agg(n="count", mean="mean", sd="std").reset_index()
               .sort_values(["model", "pct_num"]))
        g["mean"] = g["mean"].round(2)
        g["sd"] = g["sd"].round(2)
        g.to_csv("%s/sweep.csv" % out, index=False)
        print("\naccuracy vs labelled fraction (test MAE, meV):")
        piv = g.pivot(index="pct_num", columns="model", values="mean")
        print(piv.to_string())

        # per-family error at full data
        fam_cols = [c for c in sw.columns if c.startswith("mae_")
                    and c.endswith("_meV")]
        if fam_cols:
            full = sw[sw["pct_num"] == 100]
            if len(full):
                pf = full.groupby("model")[fam_cols].mean().round(2)
                pf.to_csv("%s/per_strain.csv" % out)
                print("\nper-family error at 100%% labels (meV):")
                print(pf.to_string())

    # ---- ablations ------------------------------------------------------
    ab = d[d["kind"] == "ablation"]
    if len(ab):
        g = (ab.groupby(["study", "variant", "pct_num"])["mae"]
               .agg(n="count", mean="mean", sd="std").reset_index())
        g["mean"] = g["mean"].round(2)
        g["sd"] = g["sd"].round(2)
        g.to_csv("%s/ablations.csv" % out, index=False)
        print("\n%d ablation variants collected" % g["variant"].nunique())

    print("\nwrote tables into %s" % out)


if __name__ == "__main__":
    main()
