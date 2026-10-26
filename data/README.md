# Reference dataset: `dft_hy_v9.csv`

1,292 distinct deformation states of diamond (two-atom primitive cell), one row each, with the strain tensor and the
band edges from hybrid-functional DFT. Repeated calculations of the same state in the original set of 1,340 calculations
were merged. SHA-256: `44dacb84df6a4924f236f1d42863e9b328957dd8adbc50e453c51d44eae0ac8c` (checked by `check_env.py`).
How the calculations were made: `dft/README.md` and the Methods of the paper.

| Family (`strain_type`) | States |
|---|---|
| isotropic (`isotropic`, plus the `unstrained` reference cell) | 48 |
| uniaxial (`uniaxial_x`, `_y`, `_z`) | 120 |
| biaxial (`biaxial_xy`, `_xz`, `_yz`) | 120 |
| triaxial | 512 |
| shear (one, two or three planes) | 492 |

## Columns used by the code

| Column | Meaning |
|---|---|
| `E_xx`, `E_yy`, `E_zz`, `E_xy`, `E_yz`, `E_zx` | Green–Lagrange strain tensor E = ½(FᵀF − I), relative to the PBE-relaxed unstrained cell (the model inputs) |
| `CBM_eV`, `VBM_eV` | conduction-band minimum and valence-band maximum (eV): the lowest unoccupied and highest occupied eigenvalues on the Γ-centred 7×7×7 k-point grid |
| `Eg_eV` | bandgap, `CBM_eV − VBM_eV` (eV) |
| `strain_type` | deformation family (above); the data division keeps the proportions within each family |
| `vbm_kpt_index`, `cbm_kpt_index` | index of the k-point that holds each band edge (0 = Γ) |

The other columns describe the cell (lattice vectors, volume, angles, bond length), the deformation gradient `F_*`,
derived strain invariants, the stress (GPa), convergence flags and the k-point coordinates of the band edges. They are
not used by the models.

## Notes

- The band-edge energies are Kohn–Sham eigenvalues referenced to the average electrostatic potential of each cell; only
  their difference, the gap, is independent of this reference.
- The `functional` column reads `GGA-PBE+HSE06` for every row. The hybrid step used the HSE screening (0.2 Å⁻¹) with 27%
  exact exchange (AEXX = 0.27), except the `unstrained` reference row (`id = -1`), which used 25% (see `dft/README.md`).
- For the `unstrained` row, the deformation-gradient columns `F_*` are 0 (not computed); its strain is 0.
