#!/usr/bin/env python3
"""
analyze.py — RT-NEO vibrational frequency analysis via Padé approximants.

Parses RT-NEO .out files, computes dipole spectra, detects peaks, matches
against reference frequencies, and checks convergence by splitting the
trajectory in half.

Usage:
  python analyze.py FILE [FILE ...] [options]
  python analyze.py --batch batch.txt [options]

Batch file format (one job per line, # = comment):
  path/to/file.out  1000,2000,3500
  path/to/file2.out 800,1600

Options:
  --refs F1,F2,...       Reference frequencies in cm^-1 (comma-separated)
  --window W             Ref-matching search window in cm^-1 (default: 300)
  --timestep DT          Output frame spacing in a.u. (default: 4.0)
  --sigma S              Padé damping constant in a.u. (default: 1e8)
  --w-max W              Max Padé frequency in a.u. (default: 0.1 ≈ 21950 cm^-1)
  --w-step DS            Padé frequency step in a.u. (default: 1e-6 ≈ 0.22 cm^-1)
  --save-spectra         Write per-file <name>.spectrum.dat
  --plot                 Save per-file <name>.spectrum.png
  --csv                  Write csv file <name>.csv
"""
#python analyze.py --batch batch.txt --window 2000 --csv multi.csv
import numpy as np
import re
import os
import sys
import argparse

from scipy.linalg import toeplitz, solve_toeplitz

# ── Physical constants ────────────────────────────────────────────────────────
ANGSTROM_TO_AU = 1.8897259886
AU_TO_CM1 = 27.2114 * 8065        # 219 474 cm^-1 per a.u.
ZMAP = {
    "H": 0, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}
DEFAULT_TIMESTEP = 4.0             # a.u. between RT-NEO printed frames

# ── Regex patterns ────────────────────────────────────────────────────────────
_time_re = re.compile(r"Time \(a\.u\.\):\s*([0-9Ee\+\-\.]+)")
_edip_re = re.compile(
    r"\[E\] Dipole Moment\s*\(a\.u\.\)\s*"
    r"([\-0-9Ee\.\+]+)\s+([\-0-9Ee\.\+]+)\s+([\-0-9Ee\.\+]+)"
)
_ndip_re = re.compile(
    r"\[N\] Dipole Moment\s*\(a\.u\.\)\s*"
    r"([\-0-9Ee\.\+]+)\s+([\-0-9Ee\.\+]+)\s+([\-0-9Ee\.\+]+)"
)
_pos_re = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+"
    r"([\-0-9Ee\.\+]+)\s+([\-0-9Ee\.\+]+)\s+([\-0-9Ee\.\+]+)"
)


# ── Padé approximant transform ────────────────────────────────────────────────

def pade(time, signal, sigma=1e8, max_len=None,
         w_min=0.0, w_max=0.1, w_step=1e-6,
         read_freq=None, baseline="mean"):
    """Padé approximant Fourier transform of a time-domain dipole signal.

    Args:
        time:      1-D array of sample times (a.u.)
        signal:    1-D array of signal values (a.u.)
        sigma:     damping constant (a.u.); FWHM ≈ 2/sigma
        max_len:   truncate signal to this many points before transform
        w_min/max: frequency range (a.u.)
        w_step:    frequency resolution (a.u.)
        read_freq: explicit frequency array (overrides w_min/max/step)
        baseline:  'mean' | 'first' | 'none'

    Returns:
        (fsignal, frequency) — complex spectrum and frequency array (a.u.)
    """
    signal = np.asarray(signal, dtype=float)

    if baseline == "mean":
        signal = signal - np.mean(signal)
    elif baseline == "first":
        signal = signal - signal[0]
    elif baseline != "none":
        raise ValueError(f"baseline must be 'mean', 'first', or 'none'")

    stepsize = time[1] - time[0]
    damp = np.exp(-(stepsize * np.arange(len(signal))) / sigma)
    signal = signal * damp

    M = len(signal)
    N = int(np.floor(M / 2))
    if max_len and M > max_len:
        N = int(np.floor(max_len / 2))

    d = -signal[N + 1 : 2 * N]
    try:
        b = solve_toeplitz(
            (signal[N : 2*N - 1], np.hstack([signal[1], signal[N-1 : 1 : -1]])),
            d, check_finite=False,
        )
    except Exception:
        G = signal[N + np.arange(1, N)[:, None] - np.arange(1, N)]
        b = np.linalg.solve(G, d)

    b = np.hstack([1, b])
    a = np.dot(np.tril(toeplitz(signal[:N])), b)
    p = np.poly1d(np.flip(a))
    q = np.poly1d(np.flip(b))

    frequency = np.arange(w_min, w_max, w_step) if read_freq is None else np.asarray(read_freq)
    W = np.exp(-1j * frequency * stepsize)
    return p(W) / q(W), frequency


# ── Output-file parser ────────────────────────────────────────────────────────

def parse_output(infile, quantum_atoms=None, timestep=DEFAULT_TIMESTEP):
    """Single-pass parser for RT-NEO .out files.

    Extracts the total dipole moment at each printed time step:
        D_total = D_electronic + D_classical_nuclear + D_quantum_nuclear

    Args:
        infile:        path to .out file
        quantum_atoms: set/list of 1-based atom indices treated as quantum
                       nuclei — their classical positions are excluded from
                       the nuclear dipole sum (the [N] term covers them).
                       Defaults to {1}.
        timestep:      expected time step in a.u. (used for time array only)

    Returns:
        (times, x, y, z) — 1-D numpy arrays in atomic units
    """
    if quantum_atoms is None:
        quantum_atoms = {1}
    quantum_atoms = set(quantum_atoms)

    results = []
    cur_time = None
    E_dip = N_dip = None
    nuc_pos = {}

    def _flush():
        if cur_time is None or E_dip is None:
            return
        Dn = np.zeros(3)
        for Z, R in nuc_pos.values():
            #print(Z, R)
            Dn += Z * R
        Nd = N_dip if N_dip is not None else np.zeros(3)
        #print(Nd)
        D = E_dip + Dn + Nd
        results.append((cur_time, D[0], D[1], D[2]))

    with open(infile) as fh:
        for line in fh:
            m = _time_re.search(line)
            if m:
                #print(E_dip, N_dip)
                _flush()
                cur_time = float(m.group(1))
                E_dip = N_dip = None
                nuc_pos = {}
                continue

            if cur_time is None:
                continue

            if E_dip is None:
                m = _edip_re.search(line)
                if m:
                    E_dip = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                    continue

            if N_dip is None:
                m = _ndip_re.search(line)
                if m:
                    N_dip = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                    continue

            m = _pos_re.match(line)
            
            if m:
                idx = int(m.group(1))
                #print(idx)
                sym = m.group(2)
                #### This is where Adjustments 4 and 5 are made with and idx != 1 ####
                if sym in ZMAP : #and idx != #
                    x = float(m.group(3)) * ANGSTROM_TO_AU
                    y = float(m.group(4)) * ANGSTROM_TO_AU
                    z = float(m.group(5)) * ANGSTROM_TO_AU
                    nuc_pos[idx] = (ZMAP[sym], np.array([x, y, z]))

    _flush()

    if not results:
        raise ValueError(f"No dipole data parsed from {infile}")

    arr = np.array(results)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def load_dipole_dat(filepath):
    """Read a pre-computed 'x y z' dipole file (pade.py-compatible format).

    Returns (times, x, y, z) using DEFAULT_TIMESTEP for the time array.
    """
    data = np.genfromtxt(filepath, delimiter=" ")
    n = len(data)
    times = np.arange(n) * DEFAULT_TIMESTEP
    return times, data[:, 0], data[:, 1], data[:, 2]


# ── Spectrum computation ──────────────────────────────────────────────────────

def compute_spectrum(x, y, z, timestep=DEFAULT_TIMESTEP, **pade_kwargs):
    """Run Padé on x, y, z dipole components and return combined spectrum.

    Returns:
        freqs_cm1  — frequency array in cm^-1
        spectrum   — |Sx| + |Sy| + |Sz| (total absorption proxy)
        sx, sy, sz — individual component magnitudes
    """
    n = len(x)
    t = np.arange(n, dtype=float) * timestep

    def _safe_pade(sig):
        try:
            s, w = pade(t, sig, **pade_kwargs)
            return np.abs(s), w
        except Exception:
            w = np.arange(
                pade_kwargs.get("w_min", 0.0),
                pade_kwargs.get("w_max", 0.1),
                pade_kwargs.get("w_step", 1e-6),
            )
            return np.zeros(len(w)), w

    sx, wx = _safe_pade(x)
    sy, wy = _safe_pade(y)
    sz, wz = _safe_pade(z)

    freqs_cm1 = wz * AU_TO_CM1
    return freqs_cm1, sx + sy + sz, sx, sy, sz


# ── Peak detection ────────────────────────────────────────────────────────────

def find_peaks(freqs, spectrum, threshold_frac=1e-9, eta=1):
    """Detect local maxima in the spectrum.

    Args:
        freqs:          frequency array (cm^-1)
        spectrum:       intensity array
        threshold_frac: ignore peaks below this fraction of max intensity
        eta:            neighbourhood half-width for local-max check

    Returns:
        list of (freq_cm1, intensity) sorted by frequency
    """
    if np.max(spectrum) == 0:
        return []
    threshold = threshold_frac * np.max(spectrum)
    peaks = []
    for i in range(eta, len(spectrum) - eta):
        val = spectrum[i]
        if (val > spectrum[i - 1] and val > spectrum[i + 1] and
                val > spectrum[i - eta] and val > spectrum[i + eta] and
                val > threshold):
            peaks.append((float(freqs[i]), float(val)))
    return sorted(peaks)


def _merge_peaks(peaks_list, tol=0.0001):
    """Union of per-component peak lists, deduplicating within tol cm^-1.

    When multiple components have a peak within tol of each other, keeps the
    one with the highest intensity.
    """
    all_peaks = sorted(p for peaks in peaks_list for p in peaks)
    if not all_peaks:
        return []
    merged, group = [], [all_peaks[0]]
    for peak in all_peaks[1:]:
        if peak[0] - group[0][0] <= tol:
            group.append(peak)
        else:
            merged.append(max(group, key=lambda p: p[1]))
            group = [peak]
    merged.append(max(group, key=lambda p: p[1]))
    return merged


# ── Reference matching ────────────────────────────────────────────────────────

def match_refs(peaks, refs, window=300.0):
    """Match each reference frequency to the nearest detected peak.

    Args:
        peaks:  list of (freq_cm1, intensity)
        refs:   iterable of reference frequencies in cm^-1
        window: maximum allowed |found - ref| in cm^-1

    Returns:
        list of dicts — one per ref: {ref, found, delta, intensity}
        'found' is None when no peak falls within ±window.
    """
    results = []
    for ref in refs:
        candidates = [(f, I) for f, I in peaks if abs(f - ref) <= window]
        if candidates:
            best = min(candidates, key=lambda p: abs(p[0] - ref))
            results.append({"ref": ref, "found": best[0], "delta": best[0] - ref, "intensity": best[1]})
        else:
            results.append({"ref": ref, "found": None, "delta": None, "intensity": None})
    return results


# ── Main analysis for one file ────────────────────────────────────────────────

def analyze_file(infile, refs=None, quantum_atoms=None, timestep=DEFAULT_TIMESTEP,
                 pade_kwargs=None, window=300.0, dipole_dat=False, verbose=True,
                 save_spectra=False, plot=False):
    """Full analysis pipeline for a single RT-NEO output file.

    Args:
        infile:       path to .out file (or dipole .dat file if dipole_dat=True)
        refs:         list of reference frequencies in cm^-1
        quantum_atoms: 1-based atom indices of quantum nuclei
        timestep:     frame spacing in a.u.
        pade_kwargs:  extra kwargs forwarded to pade()
        window:       reference-matching window in cm^-1
        dipole_dat:   if True, treat infile as a pre-computed x y z dipole file
        verbose:      print formatted results to stdout
        save_spectra: write <infile>.spectrum.dat
        plot:         save <infile>.spectrum.png

    Returns:
        dict with keys: file, n_steps, freqs, spectrum, sx, sy, sz,
                        peaks_full, peaks_h1, peaks_h2, ref_matches
    """
    if pade_kwargs is None:
        pade_kwargs = {}

    # ── Parse dipole time series ──────────────────────────────────────────────
    if dipole_dat:
        times, x, y, z = load_dipole_dat(infile)
    else:
        times, x, y, z = parse_output(infile, quantum_atoms=quantum_atoms, timestep=timestep)

    n = len(x)

    # ── Full-signal spectrum ──────────────────────────────────────────────────
    freqs, spec, sx, sy, sz = compute_spectrum(x, y, z, timestep=timestep, **pade_kwargs)
    peaks_full = _merge_peaks([find_peaks(freqs, sx), find_peaks(freqs, sy), find_peaks(freqs, sz)])

    # ── Convergence: 1st half vs 2nd half ────────────────────────────────────
    half= 2000
    freqs1, spec1, sx1, sy1, sz1 = compute_spectrum(x[:half], y[:half], z[:half], timestep=timestep, **pade_kwargs)
    freqs2, spec2, sx2, sy2, sz2 = compute_spectrum(x[half:], y[half:], z[half:], timestep=timestep, **pade_kwargs)
    peaks_h1 = _merge_peaks([find_peaks(freqs1, sx1), find_peaks(freqs1, sy1), find_peaks(freqs1, sz1)])
    peaks_h2 = _merge_peaks([find_peaks(freqs2, sx2), find_peaks(freqs2, sy2), find_peaks(freqs2, sz2)])

    # ── Reference matching (full signal + each half) ─────────────────────────
    ref_matches    = match_refs(peaks_full, refs or [], window=window)
    ref_matches_h1 = match_refs(peaks_h1,  refs or [], window=window)
    ref_matches_h2 = match_refs(peaks_h2,  refs or [], window=window)

    result = {
        "file":          infile,
        "n_steps":       n,
        "freqs":         freqs,
        "spectrum":      spec,
        "sx": sx, "sy": sy, "sz": sz,
        "peaks_full":    peaks_full,
        "peaks_h1":      peaks_h1,
        "peaks_h2":      peaks_h2,
        "ref_matches":   ref_matches,
        "ref_matches_h1": ref_matches_h1,
        "ref_matches_h2": ref_matches_h2,
    }

    if save_spectra:
        outpath = infile + ".spectrum.dat"
        np.savetxt(
            outpath,
            np.column_stack([freqs, spec, sx, sy, sz]),
            header="freq_cm1  total  sx  sy  sz",
            fmt="%.6f",
        )
        if verbose:
            print(f"  Saved spectrum → {outpath}")

    if plot:
        _save_plot(result, infile + ".spectrum.png")

    if verbose:
        _print_result(result, window=window)

    return result


# ── Pretty-printing ───────────────────────────────────────────────────────────

def _print_result(r, window=300.0):
    fname = os.path.basename(r["file"])
    n = r["n_steps"]
    total_t = n * DEFAULT_TIMESTEP
    print(f"\n{'='*72}")
    print(f"  {fname}   ({n} steps  |  {total_t:.0f} a.u. total)")
    print(f"{'='*72}")

    # All peaks, top 20 by intensity
    print(f"\n  All peaks (full signal, top 20 by intensity):")
    print(f"  {'Freq (cm^-1)':>14}  {'Intensity':>14}")
    top_peaks = sorted(r["peaks_full"], key=lambda p: -p[1])[:20]
    for f, I in top_peaks:
        print(f"  {f:14.1f}  {I:14.4e}")
    if not top_peaks:
        print("  (no peaks detected)")

    # Reference matches
    if r["ref_matches"]:
        print(f"\n  Reference frequency matches (window ± {window:.0f} cm^-1):")
        print(f"  {'Ref (cm^-1)':>13}  {'Found (cm^-1)':>15}  {'Δ (cm^-1)':>12}  {'Intensity':>14}")
        for m in r["ref_matches"]:
            if m["found"] is not None:
                print(f"  {m['ref']:13.1f}  {m['found']:15.1f}  {m['delta']:+12.1f}  {m['intensity']:14.4e}")
            else:
                print(f"  {m['ref']:13.1f}  {'--- no peak found ---':>28}")

    # Convergence: 1st half vs 2nd half
    print(f"\n  Convergence  (first {r['n_steps']//2} vs last {r['n_steps'] - r['n_steps']//2} steps):")
    h1_str = ", ".join(f"{f:.0f}" for f, _ in sorted(r["peaks_h1"])[:12]) or "(none)"
    h2_str = ", ".join(f"{f:.0f}" for f, _ in sorted(r["peaks_h2"])[:12]) or "(none)"
    print(f"  1st half peaks:  {h1_str}")
    print(f"  2nd half peaks:  {h2_str}")

    if r["peaks_h1"] and r["peaks_h2"]:
        top_h2 = sorted(r["peaks_h2"], key=lambda p: -p[1])[:10]
        print(f"\n  {'2nd-half (cm^-1)':>20}  {'nearest 1st-half':>20}  {'Δ (cm^-1)':>12}")
        for f2, _ in top_h2:
            f1 = min(r["peaks_h1"], key=lambda p: abs(p[0] - f2))[0]
            print(f"  {f2:20.1f}  {f1:20.1f}  {f2-f1:+12.1f}")
    print()


def _save_plot(r, outpath):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        # Full spectrum
        ax = axes[0]
        ax.plot(r["freqs"], r["spectrum"], lw=0.8, label="total")
        ax.plot(r["freqs"], r["sx"], lw=0.6, alpha=0.7, label="x")
        ax.plot(r["freqs"], r["sy"], lw=0.6, alpha=0.7, label="y")
        ax.plot(r["freqs"], r["sz"], lw=0.6, alpha=0.7, label="z")
        for f, I in r["peaks_full"]:
            ax.axvline(f, color="gray", alpha=0.3, lw=0.5)
        ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Full spectrum — {os.path.basename(r['file'])}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, r["freqs"][-1])

        # Convergence comparison
        ax2 = axes[1]
        # recompute half spectra for plotting — already in result via peaks but not stored as arrays
        ax2.set_xlabel("Frequency (cm$^{-1}$)")
        ax2.set_ylabel("Intensity (a.u.)")
        ax2.set_title("Convergence: 1st-half peaks (▲) vs 2nd-half peaks (▼)")
        for f, I in r["peaks_h1"]:
            ax2.axvline(f, color="C0", alpha=0.6, lw=1.0)
        for f, I in r["peaks_h2"]:
            ax2.axvline(f, color="C1", alpha=0.6, lw=1.0, ls="--")
        from matplotlib.lines import Line2D
        ax2.legend(handles=[
            Line2D([0], [0], color="C0", label="1st half"),
            Line2D([0], [0], color="C1", ls="--", label="2nd half"),
        ], fontsize=8)
        ax2.set_xlim(0, r["freqs"][-1])

        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"  Saved plot    → {outpath}")
    except Exception as e:
        print(f"  WARNING: could not save plot: {e}", file=sys.stderr)


# ── Summary table for batch runs ──────────────────────────────────────────────

def _print_summary(all_results):
    all_refs = sorted({m["ref"] for r in all_results for m in r["ref_matches"]})
    if not all_refs:
        return

    print(f"\n{'='*72}")
    print("  BATCH SUMMARY — closest found frequency (cm^-1) per reference")
    print(f"{'='*72}")

    col = 12
    header = f"  {'File':<34}" + "".join(f"{int(rf):>{col}}" for rf in all_refs)
    print(header)
    print("  " + "-" * (34 + col * len(all_refs)))

    for r in all_results:
        fname = os.path.basename(r["file"])[:32]
        mdict = {m["ref"]: m for m in r["ref_matches"]}
        row = f"  {fname:<34}"
        for ref in all_refs:
            m = mdict.get(ref)
            if m and m["found"] is not None:
                row += f"  {m['found']:>{col-2}.0f}"
            else:
                row += f"  {'---':>{col-2}}"
        print(row)

    # Δ rows
    print()
    header2 = f"  {'Δ = found − ref':<34}" + "".join(f"{int(rf):>{col}}" for rf in all_refs)
    print(header2)
    print("  " + "-" * (34 + col * len(all_refs)))
    for r in all_results:
        fname = os.path.basename(r["file"])[:32]
        mdict = {m["ref"]: m for m in r["ref_matches"]}
        row = f"  {fname:<34}"
        for ref in all_refs:
            m = mdict.get(ref)
            if m and m["delta"] is not None:
                row += f"  {m['delta']:>+{col-2}.0f}"
            else:
                row += f"  {'---':>{col-2}}"
        print(row)
    print()


# ── CSV export ───────────────────────────────────────────────────────────────

def _parse_name(filepath):
    """Extract method and molecule from filenames like ctpb_NCH.in.out."""
    base = os.path.basename(filepath)
    # strip extensions: ctpb_NCH.in.out → ctpb_NCH
    stem = base.split(".")[0]
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]   # method, molecule
    return stem, ""


def _write_csv(all_results, outpath):
    """Write per-reference results (full + 1st half + 2nd half) to a CSV."""
    import csv

    fieldnames = [
        "file", "method", "molecule",
        "ref_cm1",
        "full_cm1", "full_delta",
        "h1_cm1",   "h1_delta",
        "h2_cm1",   "h2_delta",
    ]

    def _val(m, key):
        v = m.get(key)
        return f"{v:.2f}" if v is not None else ""

    with open(outpath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_results:
            method, molecule = _parse_name(r["file"])
            fname = os.path.basename(r["file"])

            # zip the three match lists together (same refs, same order)
            for mf, mh1, mh2 in zip(r["ref_matches"],
                                      r["ref_matches_h1"],
                                      r["ref_matches_h2"]):
                writer.writerow({
                    "file":       fname,
                    "method":     method,
                    "molecule":   molecule,
                    "ref_cm1":    f"{mf['ref']:.2f}",
                    "full_cm1":   _val(mf,  "found"),
                    "full_delta": _val(mf,  "delta"),
                    "h1_cm1":     _val(mh1, "found"),
                    "h1_delta":   _val(mh1, "delta"),
                    "h2_cm1":     _val(mh2, "found"),
                    "h2_delta":   _val(mh2, "delta"),
                })

    print(f"\nSaved CSV → {outpath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="RT-NEO .out file(s) to analyze")
    parser.add_argument("--batch", metavar="FILE",
                        help="Batch file: each line is 'path/to/file.out  ref1,ref2,...'")
    parser.add_argument("--refs", metavar="F1,F2,...",
                        help="Reference frequencies in cm^-1 (applied to all files "
                             "that don't specify their own in a batch file)")
    parser.add_argument("--quantum-atoms", metavar="A1,A2,...", default="1",
                        help="1-based atom indices treated as quantum nuclei (default: 1)")
    parser.add_argument("--window", type=float, default=300.0,
                        help="Ref-matching search window in cm^-1 (default: 300)")
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP,
                        help=f"Frame spacing in a.u. (default: {DEFAULT_TIMESTEP})")
    parser.add_argument("--sigma", type=float, default=1e8,
                        help="Padé damping constant in a.u. (default: 1e8)")
    parser.add_argument("--w-max", type=float, default=0.1,
                        help="Max Padé frequency in a.u. (default: 0.1 ≈ 21950 cm^-1)")
    parser.add_argument("--w-step", type=float, default=1e-6,
                        help="Padé frequency step in a.u. (default: 1e-6 ≈ 0.22 cm^-1)")
    parser.add_argument("--dipole-dat", action="store_true",
                        help="Treat input files as pre-computed 'x y z' dipole files "
                             "(skips .out parsing, uses DEFAULT_TIMESTEP)")
    parser.add_argument("--csv", metavar="FILE",
                        help="Write results table (full + 1st half + 2nd half) to this CSV file")
    parser.add_argument("--save-spectra", action="store_true",
                        help="Write <file>.spectrum.dat for each input")
    parser.add_argument("--plot", action="store_true",
                        help="Save <file>.spectrum.png for each input")
    args = parser.parse_args()

    # ── Build job list ────────────────────────────────────────────────────────
    global_refs = ([float(f) for f in args.refs.split(",")]
                   if args.refs else [])
    global_qatoms = ([int(a) for a in args.quantum_atoms.split(",")]
                     if args.quantum_atoms else [1])

    jobs = []  # list of (filepath, refs_list)
    for f in args.files:
        jobs.append((f, global_refs))

    if args.batch:
        with open(args.batch) as bfh:
            for line in bfh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                path = parts[0]
                refs = ([float(r) for r in parts[1].split(",")]
                        if len(parts) > 1 else global_refs)
                jobs.append((path, refs))

    if not jobs:
        parser.print_help()
        sys.exit(1)

    pade_kwargs = {
        "sigma":  args.sigma,
        "w_max":  args.w_max,
        "w_step": args.w_step,
    }

    # ── Run analyses ──────────────────────────────────────────────────────────
    all_results = []
    for filepath, refs in jobs:
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found — skipping.", file=sys.stderr)
            continue
        try:
            r = analyze_file(
                filepath,
                refs=refs,
                quantum_atoms=global_qatoms,
                timestep=args.timestep,
                pade_kwargs=pade_kwargs,
                window=args.window,
                dipole_dat=args.dipole_dat,
                verbose=True,
                save_spectra=args.save_spectra,
                plot=args.plot,
            )
            all_results.append(r)
        except Exception as e:
            print(f"ERROR processing {filepath}: {e}", file=sys.stderr)

    # ── Batch summary table ───────────────────────────────────────────────────
    if len(all_results) > 1:
        _print_summary(all_results)

    # ── CSV export ────────────────────────────────────────────────────────────
    if args.csv and all_results:
        _write_csv(all_results, args.csv)


if __name__ == "__main__":
    main()
