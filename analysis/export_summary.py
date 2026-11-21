"""Condense the raw results of the paper into summary/, the input of make_tables.py and make_figures.py.

The raw results are not part of this repository: the run folders (training/results, about 6 GB), the VASP outputs of
the reference calculations and the preliminary studies.

Usage: python analysis/export_summary.py --dft <VASP outputs> --preliminary <preliminary studies> [--results <runs>]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_data as PD

parser = argparse.ArgumentParser(description="Write summary/ from the raw results.")
parser.add_argument("--results", default=os.path.join(REPO, "training", "results"), help="folder of the run folders")
parser.add_argument("--dft", required=True, help="folder of the VASP calculations")
parser.add_argument("--preliminary", required=True, help="folder of the preliminary studies")
args = parser.parse_args()
PD.configure(
    "raw",
    repo=REPO,
    results=os.path.abspath(args.results),
    dft=os.path.abspath(args.dft),
    preliminary=os.path.abspath(args.preliminary),
)
import metrics_report as MR

# sign accuracy: against the zero-strain gap of a cubic fit to the isotropic states with |strain| <= 0.02
data = pd.read_csv(PD.data_csv())
iso = data[data.strain_type == "isotropic"]
window = iso.E_xx.abs() <= 0.02
MR.EG0 = float(np.polyfit(iso.E_xx[window], iso.Eg_eV[window], 3)[-1])
PD.export_summary(MR.metrics, os.path.join(REPO, "summary"))
print("summary/ written")
