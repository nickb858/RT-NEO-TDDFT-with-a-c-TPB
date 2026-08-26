import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'Arial'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_xyz_blocks(filename):
    """Extract coordinate blocks from a file with repeated XYZ tables."""
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
    """Compute O–H distance in a single coordinate block (1-based indices)."""
    o = np.array(block[o_index - 1][1:])
    #print(len(block))
    h = np.array(block[h_index - 1][1:])
    return  np.linalg.norm(o - h) #h[k]

# === SETTINGS ===
files = ["ohba_sctpb.out", "ohba_otpb.out"]

linestyles = ["-", "--", ":", (0, (3, 1, 1, 1))]

donor_color = "blue"
acceptor_color = "red"

# === CREATE STACKED FIGURE ===
fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 14.5),
    sharex=True,
    gridspec_kw={"hspace": 0.0}   # flush panels
)

# Ensure axes is iterable
axes = np.atleast_1d(axes)

for i, (ax, file) in enumerate(zip(axes, files)):

    dt = 0.0096755  # fs


    blocks = parse_xyz_blocks(os.path.join(SCRIPT_DIR, file))[:-1]
    blocks = blocks[:3000]

    hind = 16

    distances   = [oh_distance(block, 7, 15, 1) for block in blocks]
    distancesq  = [oh_distance(block, 15, 9, 1) for block in blocks]

    distances2  = [oh_distance(block, hind, 7) for block in blocks]
    distances2q = [oh_distance(block, hind, 9, 1) for block in blocks]

    times = np.arange(len(blocks)) * dt

    # === PLOTS ===
    ax.plot(
        times,
        distances2,
        lw=4,
        color=donor_color,
        linestyle="-",
        label="Expectation value"
    )

    ax.plot(
        times,
        distances2q,
        lw=4,
        color=acceptor_color,
        linestyle="-"
    )

    ax.plot(
        times,
        distances,
        lw=4,
        color=donor_color,
        linestyle="--",
        label="Basis Function Center"
    )

    ax.plot(
        times,
        distancesq,
        lw=4,
        color=acceptor_color,
        linestyle="--"
    )

    # === Floating labels ===
    ax.text(
        0.12, 0.36,
        r"O$_{\rm D}$—H",
        color=donor_color,
        fontsize=30,
        transform=ax.transAxes
    )

    ax.text(
        0.12, 0.78,
        r"O$_{\rm A}$—H",
        color=acceptor_color,
        fontsize=30,
        transform=ax.transAxes
    )

    # === Panel Labels ===
    panel_label = chr(ord("A") + i)

    ax.text(
        0.015,
        0.96,
        panel_label,
        transform=ax.transAxes,
        fontsize=34,
        va="top",
        ha="left"
    )

    ax.set_ylabel("Distance (Å)", fontsize=36)

    # === Axis formatting ===
    ax.set_ylim(0.8, 2.0)

    ax.tick_params(
        axis='both',
        which='major',
        length=10,
        width=4,
        colors='black',
        labelsize=28
    )
    if i==0:
        ax.set_yticks([0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
    else:
        ax.set_yticks([0.8, 1, 1.2, 1.4, 1.6, 1.8])

    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color("black")

# Remove duplicate x tick labels on top panel
axes[0].tick_params(labelbottom=False)

# === Shared labels ===
fig.supxlabel("Time (fs)", fontsize=36)


# === Shared limits ===
axes[-1].set_xlim(0, 0.0096755 * 3000)
names = ["Expectation Value", "Basis Function Center"]
# === LEGEND ===
legend_lines = [
    Line2D(
        [0], [0],
        color="black",
        lw=5,
        linestyle=linestyles[j % len(linestyles)],
        label=names[j]
    )
    for j in range(len(names))
]

leg = axes[0].legend(
    handles=legend_lines,
    fontsize=24,
    loc="upper center",
    bbox_to_anchor=(0.75, 0.98),
    frameon=False
)

# === LAYOUT ===
plt.subplots_adjust(
    left=0.12,
    right=0.98,
    top=0.98,
    bottom=0.10,
    hspace=0.0
)

# === SAVE ===
output_file = "FigS1.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

plt.show()