import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_series(filename):
    """Load a single-column (or two-column) energy file relative to this script."""
    path = os.path.join(SCRIPT_DIR, filename)
    try:
        data = np.loadtxt(path)
    except OSError:
        print(f"Warning: '{filename}' not found, skipping.")
        return None
    data = np.array(data).squeeze()
    if data.ndim > 1:
        data = data[:, 1]
    return data


# Data files and display labels (see paper for variant descriptions)
filenames = ["ctpb.txt", "tpb_ke.txt", "tpb.txt", "fpb.txt"]
names = ["c-TPB", "Original TPB System", "Original TPB Extended", "FPB"]

# Electronic timestep in a.u. (c-TPB/TPB: DT=0.04; FPB: DT=0.4, corrected below)
dt = 0.04

styles = {
    "c-TPB":                 {"color": "purple", "linestyle": "-",  "lw": 4},
    "Original TPB System":   {"color": "black",  "linestyle": "--", "lw": 4},
    "Original TPB Extended": {"color": "black",  "linestyle": "-",  "lw": 4},
    "FPB":                   {"color": "green",  "linestyle": "-",  "lw": 4},
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 14.5), sharex=True,
                                gridspec_kw={'hspace': 0})

# --- Panel A: all methods ---
for fname, label in zip(filenames, names):
    data = load_series(fname)
    if data is None:
        continue

    t = np.arange(data.size) * dt
    data, t = data[::10], t[::10]

    if fname == "fpb.txt":
        t = t * 10.0  # FPB uses 10× larger timestep

    t_fs = t * 2.418884e-2

    ax1.plot(t_fs, (data - data[0]) * 1e3,
             label=label,
             color=styles[label]["color"],
             linestyle=styles[label]["linestyle"],
             lw=styles[label]["lw"])

ax1.set_ylabel(r"Energy Change ($10^{-3}$ a.u.)", fontsize=36)
ax1.set_xlim(0, 1200 * 2.418884e-2)
ax1.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8])
ax1.legend(fontsize=30, frameon=False, loc="best")
ax1.text(0.02, 0.90, "A", transform=ax1.transAxes, fontsize=48)

# --- Panel B: c-TPB and FPB only (zoomed energy scale) ---
for fname, label in zip(filenames, names):
    if label not in ["FPB", "c-TPB"]:
        continue

    data = load_series(fname)
    if data is None:
        continue

    t = np.arange(data.size) * dt
    data, t = data[::10], t[::10]

    if fname == "fpb.txt":
        t = t * 10.0

    t_fs = t * 2.418884e-2

    ax2.plot(t_fs, (data - data[0]) * 1e4,
             label=label,
             color=styles[label]["color"],
             linestyle=styles[label]["linestyle"],
             lw=styles[label]["lw"])

ax2.set_xlabel("Time (fs)", fontsize=36)
ax2.set_ylabel(r"Energy Change ($10^{-4}$ a.u.)", fontsize=36)
ax2.set_yticks([0, 1, 2, 3, 4, 5])
ax2.legend(fontsize=30, frameon=False, loc="best")
ax2.text(0.02, 0.90, "B", transform=ax2.transAxes, fontsize=48)

# --- Shared formatting ---
for ax in fig.axes:
    ax.tick_params(axis='both', which='major', labelsize=36, length=10, width=4,
                   colors='black')
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color("black")

fig.tight_layout()
plt.savefig("Fig3.png", dpi=300, bbox_inches="tight")
plt.show()
