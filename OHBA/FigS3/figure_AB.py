import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'Arial'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PART A helpers
# ============================================================

def parse_xyz_blocks(filename):
    blocks = []
    current = []
    in_block = False
    aft = False
    with open(filename) as f:
        for line in f:
            if "Standard Nuclear Orientation" in line:
                in_block = True
                current = []
            elif "-----" in line:
                continue
            elif in_block and line.strip():
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    _, atom, x, y, z = parts
                    current.append((atom, float(x), float(y), float(z)))
            elif in_block and not line.strip():
                if current:
                    blocks.append(current)
                in_block = False
                aft = True
            elif " [N] Dipole Moment   (a.u.)" in line:
                if aft:
                    aft = False
                    _, _, _, _, x, y, z = line.split()
                    ANGSTROM_TO_AU = 1.8897259886
                    blocks[-1].append(("H", float(x)/ANGSTROM_TO_AU, float(y)/ANGSTROM_TO_AU, float(z)/ANGSTROM_TO_AU))
    return blocks


def oh_distance(block, o_index=9, h_index=15, k=0):
    o = np.array(block[o_index - 1][1:])
    h = np.array(block[h_index - 1][1:])
    return np.linalg.norm(o - h)


# ============================================================
# PART B helpers
# ============================================================

def _load_series(filename):
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


# ============================================================
# CREATE FIGURE
# ============================================================

fig, (axA, axB) = plt.subplots(1, 2, figsize=(21, 9))

# ============================================================
# PANEL A
# ============================================================

files = ["ohba_ctpb.out", "ohba_fpb_0.4.out", "ohba_ctpb_0.4.out"]
names_A = ["c-TPB 0.04", "FPB 0.4", "c-TPB 0.4"]
linestyles = ["-", "--", ":", (0, (3, 1, 1, 1))]
donor_color = "blue"
acceptor_color = "red"

for i, (file, label) in enumerate(zip(files, names_A)):
    dt = 0.0096755
    if file in ["ohba_fpb_0.4.out", "ohba_ctpb_0.4.out"]:
        dt = dt * 10

    blocks = parse_xyz_blocks(os.path.join(SCRIPT_DIR, file))[:-1]
    blocks = blocks[:3000]

    hind = 16
    if i == 1:
        hind = 19

    distances2 = [oh_distance(block, 7, hind, 1) for block in blocks]
    distances2q = [oh_distance(block, hind, 9, 1) for block in blocks]
    times = np.arange(len(blocks)) * dt

    ls = linestyles[i % len(linestyles)]
    axA.plot(times, distances2,  lw=4, color=donor_color,    linestyle=ls, label=label)
    axA.plot(times, distances2q, lw=4, color=acceptor_color, linestyle=ls)

axA.text(0.02, 0.26, r"O$_{\rm D}$—H", color=donor_color,    fontsize=36, transform=axA.transAxes)
axA.text(0.02, 0.80, r"O$_{\rm A}$—H", color=acceptor_color, fontsize=36, transform=axA.transAxes)

legend_lines_A = [
    Line2D([0], [0], color="black", lw=4,
           linestyle=linestyles[j % len(linestyles)], label=names_A[j])
    for j in [1, 2, 0]
]
leg = axA.legend(handles=legend_lines_A, fontsize=36, loc="upper right")
for line in leg.get_lines():
    line.set_color('black')

axA.set_ylabel("Distance (Å)", fontsize=40)
axA.set_xlabel("Time (fs)", fontsize=40)
axA.set_xlim(0, 0.0096755 * 3000)
axA.set_xticks(range(0, 30, 5))
axA.set_ylim(0.9, 2.1)
axA.tick_params(axis='both', which='major', labelsize=36, length=10, width=4, colors='black')
for spine in axA.spines.values():
    spine.set_linewidth(4)
    spine.set_color("black")

# ============================================================
# PANEL B
# ============================================================

filenames_B = ["fpb_0.4.txt", "ctpb_0.4.txt", "ctpb_0.04.txt"]
names_B = ["FPB 0.4", "c-TPB 0.4", "c-TPB 0.04"]
dt_B = 0.04

styles = {
    "c-TPB 0.4":  {"color": "purple", "linestyle": "-",  "lw": 4},
    "c-TPB 0.04": {"color": "blue",   "linestyle": "-",  "lw": 4},
    "Original TPB System":   {"color": "black",  "linestyle": "--", "lw": 4},
    "Original TPB Extended": {"color": "black",  "linestyle": "-",  "lw": 4},
    "FPB 0.4":    {"color": "green",  "linestyle": "-",  "lw": 4},
}

for fname, label in zip(filenames_B, names_B):
    data = _load_series(fname)
    if data is None:
        continue
    t = np.arange(data.size) * dt_B
    data = data[::10]
    t = t[::10]
    if fname in ["fpb_0.4.txt", "ctpb_0.4.txt"]:
        t = t * 10.0
    t_fs = t * 2.418884e-2
    axB.plot(t_fs, (data - data[0]) * 1e4,
             label=label,
             color=styles[label]["color"],
             linestyle=styles[label]["linestyle"],
             lw=styles[label]["lw"])

axB.set_xlabel("Time (fs)", fontsize=36)
axB.set_ylabel(r"Energy Change ($10^{-4}$ a.u.)", fontsize=36, labelpad=-2)
axB.set_xlim(0, 1200 * 2.418884e-2)
axB.set_xticks(range(0, 30, 5))
axB.set_yticks([-10, -5, 0, 5, 10])
axB.legend(fontsize=30, frameon=False, loc="best")
axB.tick_params(axis='both', which='major', labelsize=36, length=10, width=4, colors='black')
for spine in axB.spines.values():
    spine.set_linewidth(4)
    spine.set_color("black")

# ============================================================
# PANEL LABELS
# ============================================================

for ax, label in [(axA, "A"), (axB, "B")]:
    ax.text(-0.12, 1.02, label,
            transform=ax.transAxes,
            fontsize=48, fontweight='bold',
            va='bottom', ha='left')

# ============================================================
# SAVE
# ============================================================

fig.tight_layout(w_pad=4)
plt.savefig("OHBA_AB.png", dpi=300, bbox_inches="tight")
plt.show()
