"""Check the dataset: row count, columns, missing values, deformation family mix and
duplicate strain states.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def family_of(label):
    """Collapse the CSV's sub-labels (uniaxial_x, unstrained, ...) to 5 families."""
    s = str(label).lower().strip()
    if s.startswith("uniaxial"):
        return "uniaxial"
    if s.startswith("biaxial"):
        return "biaxial"
    if s in ("isotropic", "triaxial", "shear"):
        return s
    return "isotropic"          # 'unstrained' counts as the isotropic reference


def main():
    df = pd.read_csv(C.DATA_CSV)
    out = C.results_dir("data_checks")

    # ---- structure ------------------------------------------------------
    needed = C.STRAIN_COLS + [C.TARGET_EG, C.TARGET_CBM, C.TARGET_VBM]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit("dataset is missing required columns: %s" % missing)

    n_null = int(df[needed].isnull().sum().sum())
    print("rows: %d    required columns: all present    missing values: %d"
          % (len(df), n_null))
    if n_null:
        raise SystemExit("dataset has missing values -- stop and fix the CSV")

    # ---- composition ----------------------------------------------------
    fam = df[C.TARGET_EG].groupby(df["strain_type"].map(family_of))
    rows = []
    print("\n%-11s %6s  %18s  %18s" % ("family", "n", "Eg range (eV)", "Eg mean (eV)"))
    for name in C.STRAIN_FAMILIES:
        if name not in fam.groups:
            continue
        v = fam.get_group(name)
        rows.append(dict(family=name, n=len(v), eg_min=round(v.min(), 4),
                         eg_max=round(v.max(), 4), eg_mean=round(v.mean(), 4)))
        print("%-11s %6d  %8.3f - %-8.3f %12.4f"
              % (name, len(v), v.min(), v.max(), v.mean()))
    print("%-11s %6d" % ("TOTAL", len(df)))

    # ---- strain magnitudes ----------------------------------------------
    s = df[C.STRAIN_COLS].values
    print("\nstrain components: min %.4f  max %.4f  (dimensionless Green-Lagrange)"
          % (s.min(), s.max()))

    # ---- repeated strain states -----------------------------------------
    # Some strain vectors appear more than once. Where they do, the labels are
    # identical too, so these are the same calculation stored repeatedly rather
    # than a reproducibility probe. It matters because the split is by ROW: two
    # copies of one state can land on opposite sides of the train/test boundary,
    # and the test copy is then trivially predictable.
    key = df[C.STRAIN_COLS].round(9).apply(tuple, axis=1)
    counts = key.value_counts()
    repeated = counts[counts > 1]
    n_rep_rows = int(repeated.sum())
    print("\ndistinct strain states: %d of %d rows" % (key.nunique(), len(df)))
    if len(repeated):
        def spread(values):
            """Range of the bandgap within one group of duplicate states."""
            return values.max() - values.min()

        lab_spread = df.groupby(key)[C.TARGET_EG].agg(spread)
        print("repeated states: %d, covering %d rows (%.1f%%); copies per state %s"
              % (len(repeated), n_rep_rows, 100.0 * n_rep_rows / len(df),
                 sorted(repeated.unique())))
        print("largest label disagreement between copies: %.3f meV"
              % (lab_spread.max() * 1000.0))

    # ---- split integrity -------------------------------------------------
    stype = df["strain_type"].map(family_of).values
    idx = np.arange(len(df))
    twins = []
    for seed in C.SEEDS:
        i_tv, i_te = train_test_split(idx, test_size=C.TEST_FRACTION,
                                      random_state=seed, stratify=stype)
        i_tr, _ = train_test_split(i_tv, test_size=C.VAL_FRACTION_OF_REMAINDER,
                                   random_state=seed, stratify=stype[i_tv])
        train_keys = set(key.iloc[i_tr])
        twins.append(sum(1 for k in key.iloc[i_te] if k in train_keys))
    n_te = len(i_te)
    print("\ntest rows holding an exact copy of a training row: "
          "mean %.1f of %d (%.1f%%), range %d-%d over %d seeds"
          % (np.mean(twins), n_te, 100.0 * np.mean(twins) / n_te,
             min(twins), max(twins), len(C.SEEDS)))
    print("  -> every model is scored on the same rows, so comparisons are fair,")
    print("     but absolute errors are slightly optimistic. 15_ablate_duplicates.py")
    print("     re-runs the headline models on distinct states only.")

    summary = pd.DataFrame(rows)
    summary.to_csv("%s/dataset_summary.csv" % out, index=False)
    pd.DataFrame(dict(metric=["rows", "distinct_states", "repeated_states",
                              "repeated_rows", "test_rows_with_train_copy_mean"],
                      value=[len(df), key.nunique(), len(repeated), n_rep_rows,
                             round(float(np.mean(twins)), 1)])).to_csv(
        "%s/integrity_summary.csv" % out, index=False)
    print("\nwrote %s/dataset_summary.csv and integrity_summary.csv" % out)


if __name__ == "__main__":
    main()
