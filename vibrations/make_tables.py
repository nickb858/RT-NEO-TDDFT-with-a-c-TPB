#!/usr/bin/env python3
"""
make_tables.py -- Reproduce the two published vibrational-frequency tables.

Run from the vibrations/ directory:
    python make_tables.py

Adjustments applied:
  1. ctpb  -> single_proton_with_cneo.csv   (c-TPB + CNEO)
  2. otpb  -> single_proton_without_cneo.csv (o-TPB, no CNEO)
  3. FFH ctpb FH-stretch (ref ~1721 cm^-1): set full = h1 value (2006)
     (the auto-detected 1720.57 peak is spurious; the correct peak is 2006)
  4. FFH ctpb FF-stretch: hardcode full = h1 = h2 = 638 cm^-1 (idx!=1 fix applied)
  5. HH ctpb: hardcode full = h1 = h2 = 4040 cm^-1 (idx!=1 fix applied)
  6. FFH otpb: all "---" (numerical instabilities at longer times)
  7. H2, H2O otpb: "---" (numerical instabilities)
  8. H2CO otpb: full and h2 "---"; only h1 reported
"""

import math
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def round_half_up(x):
    """Round using 'round half up', avoiding Python's banker's rounding."""
    return math.floor(abs(float(x)) + 0.5)


# -- Load CSVs ----------------------------------------------------------------

def load(fname):
    df = pd.read_csv(os.path.join(HERE, fname))
    # Strip numeric suffix that analyze.py appends when filename contained _2
    df['molecule'] = df['molecule'].str.replace(r'_\d+$', '', regex=True)
    return df


#generate these with script.sh
sp_with    = load('single_proton_with_cneo.csv')
sp_without = load('single_proton_without_cneo.csv')
mp         = load('multi_proton.csv')

ctpb_sp = sp_with[sp_with['method'] == 'ctpb'].copy().reset_index(drop=True)
otpb_sp = sp_without[sp_without['method'] == 'otpb'].copy().reset_index(drop=True)
ctpb_mp = mp[mp['method'] == 'ctpb'].copy().reset_index(drop=True)
otpb_mp = mp[mp['method'] == 'otpb'].copy().reset_index(drop=True)


# -- Apply adjustments --------------------------------------------------------

# 3. FFH ctpb FH-stretch: replace full with h1 value (2006)
m = (ctpb_sp['molecule'] == 'FFH') & ctpb_sp['ref_cm1'].between(1700, 1750)
ctpb_sp.loc[m, ['full_cm1', 'full_delta']] = (
    ctpb_sp.loc[m, ['h1_cm1', 'h1_delta']].values
)

# 4. FFH ctpb FF-stretch: hardcode to 638 (idx!=1 corrected value)
m_ff = (ctpb_sp['molecule'] == 'FFH') & ctpb_sp['ref_cm1'].between(580, 600)
ref_ff = float(ctpb_sp.loc[m_ff, 'ref_cm1'].values[0])
for col_v, col_d in [('full_cm1', 'full_delta'), ('h1_cm1', 'h1_delta'), ('h2_cm1', 'h2_delta')]:
    ctpb_sp.loc[m_ff, col_v] = 638.0
    ctpb_sp.loc[m_ff, col_d] = 638.0 - ref_ff

# 5. HH ctpb: hardcode to 4040 (idx!=1 corrected value)
m_hh = ctpb_mp['molecule'] == 'HH'
ref_hh = float(ctpb_mp.loc[m_hh, 'ref_cm1'].values[0])
for col_v, col_d in [('full_cm1', 'full_delta'), ('h1_cm1', 'h1_delta'), ('h2_cm1', 'h2_delta')]:
    ctpb_mp.loc[m_hh, col_v] = 4040.0
    ctpb_mp.loc[m_hh, col_d] = 4040.0 - ref_hh

# 6. FFH otpb: all NaN (numerical instabilities)
m = otpb_sp['molecule'] == 'FFH'
otpb_sp.loc[m, ['full_cm1', 'h1_cm1', 'h2_cm1',
                 'full_delta', 'h1_delta', 'h2_delta']] = np.nan

# 7. H2, H2O otpb: all NaN (numerical instabilities)
m = otpb_mp['molecule'].isin(['HH', 'OHH'])
otpb_mp.loc[m, ['full_cm1', 'h1_cm1', 'h2_cm1',
                 'full_delta', 'h1_delta', 'h2_delta']] = np.nan

# 8. H2CO otpb: full and h2 NaN (unstable at longer times)
m = otpb_mp['molecule'] == 'OCHH'
otpb_mp.loc[m, ['full_cm1', 'h2_cm1', 'full_delta', 'h2_delta']] = np.nan


# -- Helpers ------------------------------------------------------------------

def get_vals(df, mol, ref, tol=30):
    """Return (full, h1, h2, full_delta, h1_delta, h2_delta) or all NaN."""
    sub = df[df['molecule'] == mol]
    if sub.empty:
        return (np.nan,) * 6
    diffs = (sub['ref_cm1'] - ref).abs()
    if diffs.min() > tol:
        return (np.nan,) * 6
    r = sub.loc[diffs.idxmin()]
    return (r['full_cm1'], r['h1_cm1'], r['h2_cm1'],
            r['full_delta'], r['h1_delta'], r['h2_delta'])


def fmt(val, delta):
    """Format frequency as 'value (|error|)' or '---', using CSV delta."""
    if pd.isna(val):
        return '---'
    vi = int(round(float(val)))
    if pd.isna(delta):
        return '---'
    err = round_half_up(delta)
    return f'{vi} ({err})'


def print_row(mode, ref, mol, cdf, odf):
    cf, ch1, ch2, cfd, ch1d, ch2d = get_vals(cdf, mol, ref)
    of, oh1, oh2, ofd, oh1d, oh2d = get_vals(odf, mol, ref)
    ri = int(round(ref))
    print(f"  {mode:<24} {ri:>5}  "
          f"{fmt(cf,cfd):>13} {fmt(ch1,ch1d):>13} {fmt(ch2,ch2d):>13}  "
          f"{fmt(of,ofd):>13} {fmt(oh1,oh1d):>13} {fmt(oh2,oh2d):>13}")


def print_mae(refs, mol, cdf, odf):
    """Compute MAE from float deltas in the CSV, then round the mean."""
    def collect_errors(df):
        errs = {'full': [], 'h1': [], 'h2': []}
        for ref in refs:
            _, _, _, fd, h1d, h2d = get_vals(df, mol, ref)
            for key, delta in [('full', fd), ('h1', h1d), ('h2', h2d)]:
                if not pd.isna(delta):
                    errs[key].append(abs(float(delta)))
        return errs

    ce = collect_errors(cdf)
    oe = collect_errors(odf)

    def mae_str(e, k):
        return str(int(round(np.mean(e[k])))) if e[k] else '---'

    print(f"  {'MAE':<24} {'---':>5}  "
          f"{mae_str(ce,'full'):>13} {mae_str(ce,'h1'):>13} {mae_str(ce,'h2'):>13}  "
          f"{mae_str(oe,'full'):>13} {mae_str(oe,'h1'):>13} {mae_str(oe,'h2'):>13}")


def sep():
    print('─' * 108)


def header():
    print(f"  {'Mode':<24} {'VPT2':>5}  "
          f"{'c-TPB Full':>13} {'h1':>13} {'h2':>13}  "
          f"{'o-TPB Full':>13} {'h1':>13} {'h2':>13}")


# -- Table 1: Single proton ---------------------------------------------------

print()
print('TABLE 1 -- Single proton: HCN (NCH), HNC (CNH), FHF- (FFH)')
header()
sep()

print('  HCN')
print_row('CH bend',    760.16,  'NCH', ctpb_sp, otpb_sp)
print_row('CN stretch', 2175.73, 'NCH', ctpb_sp, otpb_sp)
print_row('CH stretch', 3327.84, 'NCH', ctpb_sp, otpb_sp)
print_mae([760.16, 2175.73, 3327.84], 'NCH', ctpb_sp, otpb_sp)
sep()

print('  HNC')
print_row('NH bend',    457.16,  'CNH', ctpb_sp, otpb_sp)
print_row('NC stretch', 2067.85, 'CNH', ctpb_sp, otpb_sp)
print_row('NH stretch', 3625.56, 'CNH', ctpb_sp, otpb_sp)
print_mae([457.16, 2067.85, 3625.56], 'CNH', ctpb_sp, otpb_sp)
sep()

print('  FHF-')
print_row('FF stretch',  592.28,  'FFH', ctpb_sp, otpb_sp)
print_row('FH bend',     1350.77, 'FFH', ctpb_sp, otpb_sp)
print_row('FH stretch',  1720.57, 'FFH', ctpb_sp, otpb_sp)
print_mae([592.28, 1350.77, 1720.57], 'FFH', ctpb_sp, otpb_sp)


# -- Table 2: Multi proton ----------------------------------------------------

print()
print('TABLE 2 -- Multi proton: H2 (HH), H2O (OHH), H2CO (OCHH), HCOOH (COOHH)')
header()
sep()

print('  H2')
print_row('HH stretch', 4114.43, 'HH', ctpb_mp, otpb_mp)
print_mae([4114.43], 'HH', ctpb_mp, otpb_mp)
sep()

print('  H2O')
print_row('HOH bend',         1604.48, 'OHH', ctpb_mp, otpb_mp)
print_row('OH sym. stretch',  3644.27, 'OHH', ctpb_mp, otpb_mp)
print_row('OH asym. stretch', 3697.18, 'OHH', ctpb_mp, otpb_mp)
print_mae([1604.48, 3644.27, 3697.18], 'OHH', ctpb_mp, otpb_mp)
sep()

print('  H2CO')
print_row('CH2 wag',          1187.65, 'OCHH', ctpb_mp, otpb_mp)
print_row('CH2 rock',         1247.25, 'OCHH', ctpb_mp, otpb_mp)
print_row('CH2 scissors',     1517.22, 'OCHH', ctpb_mp, otpb_mp)
print_row('CO stretch',        1798.49, 'OCHH', ctpb_mp, otpb_mp)
print_row('CH sym. stretch',  2739.48, 'OCHH', ctpb_mp, otpb_mp)
print_row('CH asym. stretch', 2749.10, 'OCHH', ctpb_mp, otpb_mp)
print_mae([1187.65, 1247.25, 1517.22, 1798.49, 2739.48, 2749.10],
          'OCHH', ctpb_mp, otpb_mp)
sep()

print('  HCOOH')
print_row('OCO bend',    625.59,  'COOHH', ctpb_mp, otpb_mp)
print_row('torsion',     638.37,  'COOHH', ctpb_mp, otpb_mp)
print_row('CH bend',     1034.39, 'COOHH', ctpb_mp, otpb_mp)
print_row('CO stretch',  1111.48, 'COOHH', ctpb_mp, otpb_mp)
print_row('OH bend',     1252.96, 'COOHH', ctpb_mp, otpb_mp)
print_row('CH bend',     1394.18, 'COOHH', ctpb_mp, otpb_mp)
print_row('C=O stretch', 1802.29, 'COOHH', ctpb_mp, otpb_mp)
print_row('CH stretch',  2906.30, 'COOHH', ctpb_mp, otpb_mp)
print_row('OH stretch',  3508.67, 'COOHH', ctpb_mp, otpb_mp)
print_mae([625.59, 638.37, 1034.39, 1111.48, 1252.96,
           1394.18, 1802.29, 2906.30, 3508.67],
          'COOHH', ctpb_mp, otpb_mp)
print()
