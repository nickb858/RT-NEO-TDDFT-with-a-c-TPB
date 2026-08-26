# RT-NEO-TDDFT with c-TPB

Scripts and Q-Chem input/output files for reproducing the figures and tables in the manuscript
describing the **corrected Traveling Proton Basis (c-TPB)** method for real-time nuclear-electronic
orbital TDDFT (RT-NEO-TDDFT).

The paper compares c-TPB against the original TPB (o-TPB) and the Fixed Proton Basis (FPB) for
two systems:
- **OHBA** — o-hydroxybenzaldehyde (intramolecular proton transfer)
- **BPOH2** — \[2,2'-bipyridyl\]-3,3'-diol (double intramolecular proton transfer)

Vibrational frequencies are benchmarked for single- and multi-proton molecules (H₂, H₂O, H₂CO,
HCOOH, HCN, HNC, FHF⁻) against VPT2 references.

---

## Dependencies

- Python ≥ 3.8
- NumPy
- SciPy (vibrational analysis only)
- Matplotlib
- pandas (table printing only)

Install with:
```
pip install numpy scipy matplotlib pandas
```

---

## Repository Layout

```
OHBA/
  Fig2/        O–H distance vs time (3 methods: c-TPB, o-TPB, FPB)
  Fig3/        Energy conservation (all methods; two zoom levels)
  FigS1/       O–H distance: proton expectation value vs basis-center position
  FigS2/       2-D proton trajectory scatter + molecule inset
  FigS3/       O–H distance and energy conservation at different timesteps

BPOH2/
  Fig4/        O–H and N–H distances vs time (c-TPB)
  Fig5/        Energy conservation (c-TPB)

vibrations/
  analyze.py           Dipole-spectrum analysis via Padé approximants
  make_tables.py       Print Table 1 (single proton) and Table 2 (multi proton)
  script.sh            Run analyze.py on all batch files → CSV outputs
  single_proton/
    batch_with_cneo.txt     Jobs with CNEO correction
    batch_without_cneo.txt  Jobs without CNEO correction
    with_cneo/              Q-Chem output files (c-TPB and o-TPB, with CNEO)
    without_cneo/           Q-Chem output files (c-TPB and o-TPB, without CNEO)
    VPT2/                   VPT2 reference frequency files
  multi_proton/
    batch.txt           Jobs for multi-proton molecules
    outputs/            Q-Chem output files (c-TPB and o-TPB)
    VPT2/               VPT2 reference frequency files
```

> **Large files not included:** `BPOH2/Fig4/BPOH2_ctpb.out` (130 MB) is not committed.
> `BPOH2/Fig4/BPOH2_ctpb_distances.csv` (pre-extracted O–H and N–H distances) is provided so
> `Fig4/analysis.py` runs without the full output file.

> **Q-Chem inputs:** Each `.out` file embeds the complete Q-Chem input at the top of the file.
> The exact calculation settings — keywords, molecular geometry, and basis set — for every job
> can be recovered from the corresponding `.out` file without needing a separate `.in` file.

---

## Reproducing Figures

Each script can be run from any working directory.
Outputs are written next to the script.

### OHBA — Figure 2

O–H (donor) and O–H (acceptor) distance vs time for c-TPB, o-TPB, and FPB.

**Inputs** (in `OHBA/Fig2/`): `ohba_tpb.out`, `ohba_otpb.out`, `ohba_fpb.out`

```bash
python OHBA/Fig2/analysis.py
# → OHBA/Fig2/Fig2.png
```

---

### OHBA — Figure 3

Total energy conservation for all methods (Panel A: 10⁻³ a.u. scale;
Panel B: c-TPB and FPB at 10⁻⁴ a.u. scale).

**Inputs** (in `OHBA/Fig3/`): `ctpb.txt`, `tpb_ke.txt`, `tpb.txt`, `fpb.txt`
(single-column total-energy time series in a.u.)

```bash
python OHBA/Fig3/plot.py
# → OHBA/Fig3/Fig3.png
```

---

### OHBA — Figure S1

Stacked two-panel comparison of the quantum proton O–H distance as tracked by the
expectation value (solid) vs the basis-function center (dashed) for sc-TPB (Panel A)
and o-TPB (Panel B).

**Inputs** (in `OHBA/FigS1/`): `ohba_sctpb.out`, `ohba_otpb.out`

```bash
python OHBA/FigS1/analysis.py
# → OHBA/FigS1/FigS1.png
```

---

### OHBA — Figure S2

2-D scatter of quantum proton trajectories (c-TPB in blue, FPB in red) with fixed
proton basis-function centers and a molecule-structure inset.

**Inputs** (in `OHBA/FigS2/`): `ohba_ctpb.out`, `ohba_fpb.out`, `molecule.png`

```bash
python OHBA/FigS2/traj.py
# → OHBA/FigS2/FigS2.png
```

---

### OHBA — Figure S3

Side-by-side panels:  
- **Panel A** — O–H distance vs time for c-TPB (dt = 0.04 a.u.), FPB (dt = 0.4 a.u.),
  and c-TPB (dt = 0.4 a.u.)  
- **Panel B** — Energy conservation for the same three variants

**Inputs** (in `OHBA/FigS3/`):
- Panel A: `ohba_ctpb.out`, `ohba_fpb_0.4.out`, `ohba_ctpb_0.4.out`
- Panel B: `fpb_0.4.txt`, `ctpb_0.4.txt`, `ctpb_0.04.txt`

```bash
python OHBA/FigS3/figure_AB.py
# → OHBA/FigS3/OHBA_AB.png
```

---

### BPOH2 — Figure 4

O–H (donor, blue) and N–H (acceptor, red) distances vs time for the BPOH2
bimolecular proton transfer with c-TPB.

**Inputs** (in `BPOH2/Fig4/`): `BPOH2_ctpb_distances.csv` (included).  
If this CSV is absent the script falls back to parsing `BPOH2_ctpb.out` (not included;
place it in the same directory and the CSV will be generated automatically).

```bash
python BPOH2/Fig4/analysis.py
# → BPOH2/Fig4/Fig4.png
```

---

### BPOH2 — Figure 5

Total energy conservation for the BPOH2 c-TPB simulation.

**Inputs** (in `BPOH2/Fig5/`): `ctpb.txt` (single-column total-energy series)

```bash
python BPOH2/Fig5/plot.py
# → BPOH2/Fig5/Fig5.png
```

---

## Reproducing Tables (Vibrational Frequencies)

Tables 1 and 2 report vibrational frequencies (cm⁻¹) from dipole spectra computed via
Padé approximants and compared against VPT2 references.

### Step 1 — Generate CSV files

Run from the repo root:

```bash
cd vibrations
bash script.sh
```

This is equivalent to:

```bash
python analyze.py --batch single_proton/batch_with_cneo.txt    --window 1000 --csv single_proton_with_cneo.csv
python analyze.py --batch single_proton/batch_without_cneo.txt --window 1000 --csv single_proton_without_cneo.csv
python analyze.py --batch multi_proton/batch.txt               --window 1000 --csv multi_proton.csv
```

Each batch file lists Q-Chem output paths and comma-separated VPT2 reference frequencies.
The three CSV files are written to `vibrations/`.

### Step 2 — Print the tables

```bash
python vibrations/make_tables.py
```

Prints Table 1 (single proton: HCN, HNC, FHF⁻) and Table 2 (multi proton: H₂, H₂O,
H₂CO, HCOOH) to stdout, with known corrections applied (see comments in `make_tables.py`).

### analyze.py options

`analyze.py` can also be run on individual files:

```bash
python vibrations/analyze.py path/to/file.out --refs 760,2175,3327 --plot --save-spectra
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--refs F1,F2,...` | — | Reference frequencies in cm⁻¹ |
| `--window W` | 300 | Peak-matching search window (cm⁻¹) |
| `--timestep DT` | 4.0 | Output frame spacing (a.u.) |
| `--sigma S` | 1e8 | Padé damping constant (a.u.) |
| `--w-max W` | 0.1 | Max frequency (a.u.; ≈ 21 950 cm⁻¹) |
| `--csv FILE` | — | Write results to CSV |
| `--plot` | off | Save spectrum PNG |
| `--save-spectra` | off | Write spectrum `.dat` files |
| `--quantum-atoms A,...` | — | Restrict nuclear dipole to listed atom indices |
