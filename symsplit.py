"""Symmetry-grouped train/validation/test split, shared by the baseline trainer (the PINN trainer carries the same code
inline). Strains related by one of the 48 cubic operations carry the same labels, so each symmetry class goes to one
partition; classes are drawn per family (70/10/20 of the family's states) with the given seed."""
import itertools
import numpy as np


def symmetry_split(strain_all, stype_all, seed):
    ops = [np.diag(sg) @ np.eye(3)[list(pm)] for pm in itertools.permutations(range(3)) for sg in itertools.product([1, -1], repeat=3)]
    keys = []
    for v in strain_all.astype(np.float64):
        T = np.array([[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]])
        imgs = []
        for R in ops:
            Tr = R @ T @ R.T
            imgs.append(tuple(np.round([Tr[0, 0], Tr[1, 1], Tr[2, 2], Tr[0, 1], Tr[1, 2], Tr[0, 2]], 5)))
        keys.append(min(imgs))
    uniq = {}
    cls = np.array([uniq.setdefault(k, len(uniq)) for k in keys])
    idx_all = np.arange(len(strain_all)); rng_s = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for fam in np.unique(stype_all):
        fam_idx = idx_all[stype_all == fam]; fam_cls = np.unique(cls[fam_idx]); rng_s.shuffle(fam_cls)
        n = len(fam_idx); te_n, va_n = 0, 0
        for c in fam_cls:
            members = fam_idx[cls[fam_idx] == c]
            if te_n < round(0.20 * n): idx_te.extend(members); te_n += len(members)
            elif va_n < round(0.10 * n): idx_va.extend(members); va_n += len(members)
            else: idx_tr.extend(members)
    return np.array(sorted(idx_tr)), np.array(sorted(idx_va)), np.array(sorted(idx_te))
