# Reference calculations: VASP inputs and relaxed structures

These files cover the reference calculations of the paper: the input settings, the atomic coordinates of every
relaxed structure and the absolute energies.

| File | Content |
|---|---|
| `relaxed_structures_and_absolute_energies.csv` | One row per state (1,292): lattice vectors (Å) and fractional coordinates of the two carbon atoms after the PBE relaxation, the cell volume, and the energies from the hybrid-functional step: total energy of the cell, valence-band maximum, conduction-band minimum, gap and Fermi level (eV). `source` gives the original calculation folder. |
| `contcar_relaxed_PBE.tar.gz` | The same structures as VASP CONTCAR files, one per state (`structures/state_<id>_<name>.CONTCAR`). |
| `INCAR_1-GGA-PBE_relaxation` | Step 1: relaxation of the atomic positions at fixed cell shape and volume with the PBE functional (IBRION = 2, ISIF = 2, EDIFFG = −0.001 eV/Å, NSW = 500). |
| `INCAR_2-HSE_single_point` | Step 2: single-point HSE calculation on the relaxed structure (LHFCALC, HFSCREEN = 0.2 Å⁻¹, AEXX = 0.27, i.e. 27% exact exchange), started from the step-1 wavefunctions and charge density (ISTART = 1, ICHARG = 1). |
| `KPOINTS` | Γ-centred 7×7×7 grid, identical for both steps and all states. |
| `POTCAR.spec` | Identifier of the PAW potential (PAW_PBE C 08Apr2002, four valence electrons). The potential file itself is VASP-licensed and is not redistributed. |

Both steps use ENCUT = 550 eV, Gaussian smearing of 0.01 eV, no spin polarization, EDIFF = 10⁻⁶ eV and no symmetry
(ISYM = 0). Code: VASP 6.2.0. The full list of settings is Supplementary Table 11 of the paper.

Notes:
- `INCAR_1` contains `PREC = Accuratei` (a stray character in the original input). VASP reads the setting as
  Accurate (`PREC = accura` in every OUTCAR), as reported in the paper.
- The band edges are the highest occupied and lowest unoccupied eigenvalues on the 7×7×7 grid. The valence-band maximum
  lies at Γ in every state. The conduction-band minimum of diamond lies between grid points, so the value is the lowest
  conduction eigenvalue among the sampled points (Supplementary Note 6 of the paper).
- The unstrained reference cell (`id = -1`, `zero_strain_reference`) was computed with 25% exact exchange (AEXX = 0.25),
  unlike the 1,291 strained states (27%). Its gap is 5.409 eV. The paper does not use this row as the zero-strain
  reference: the unstrained gap (5.50 eV) and the hydrostatic deformation potential are taken from a cubic fit to the
  isotropic states with |ε| ≤ 0.02, which all used 27%. In training and testing the row is one of the 48 isotropic
  states.
- Strain tensors in `data/dft_hy_v9.csv` are Green–Lagrange tensors E = ½(FᵀF − I), computed from these relaxed
  lattice vectors relative to the PBE-relaxed unstrained cell (lattice constant 3.574 Å, volume 11.413 Å³). The
  non-isotropic cells were generated from the experimental lattice constant (3.567 Å) and the isotropic ones from the
  relaxed cell, so the recorded non-isotropic strains include the 0.2% difference between the two lattice constants.
