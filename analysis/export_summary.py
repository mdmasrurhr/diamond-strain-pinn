#!/usr/bin/env python3
"""Condense the raw run folders of the paper into summary/ (the input of make_tables.py and make_figures.py).

The raw folders (about 18 GB: one folder per training run, the DFT outputs and the preliminary studies) are not part
of this repository. This script records how summary/ was made: paper_data.export_summary reads them in the campaign's
folder layout and writes every file of summary/.
Usage: python analysis/export_summary.py <DiamondPINN_MASTER folder>"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as PD

PD.configure("raw", master=os.path.abspath(sys.argv[1]))
import metrics_report as MR

# sign accuracy is measured against the gap at zero strain from a cubic fit to the isotropic states (as in make_tables.py)
data = pd.read_csv(PD.data_csv())
iso = data[data.strain_type == "isotropic"]
window = iso.E_xx.abs() <= 0.02
MR.EG0 = float(np.polyfit(iso.E_xx[window], iso.Eg_eV[window], 3)[-1])
PD.export_summary(MR.metrics, os.path.join(HERE, "..", "summary"))
print("summary/ written")
