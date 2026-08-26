import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

mpl.rcParams['font.family'] = 'Arial'

ANGSTROM_TO_AU = 1.8897259886


def parse_proton_traj(filename):
    """Extract proton expectation value (x, y) in Angstroms from [N] Dipole Moment lines."""
    xs, ys = [], []
    with open(filename) as f:
        for line in f:
            if "[N] Dipole Moment" in line:
                parts = line.split()
                xs.append(float(parts[4]) / ANGSTROM_TO_AU)
                ys.append(float(parts[5]) / ANGSTROM_TO_AU)
    return np.array(xs), np.array(ys)


def parse_basis_centers(filename):
    """Extract all fixed proton basis centers in Angstroms from the $molecule input block.

    Includes the quantum H atom immediately preceding the Gh block (which acts as a
    basis center but carries no Gh label) plus all Gh ghost atoms.
    """
    mol_lines = []
    in_molecule = False
    with open(filename) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "$molecule":
                in_molecule = True
                continue
            elif stripped == "$end" and in_molecule:
                break
            elif in_molecule:
                mol_lines.append(stripped)

    gx, gy = [], []
    first_gh_idx = None
    for i, line in enumerate(mol_lines):
        if line.startswith("Gh"):
            if first_gh_idx is None:
                first_gh_idx = i
            parts = line.split()
            gx.append(float(parts[1]))
            gy.append(float(parts[2]))

    # The H atom immediately before the Gh block is also a basis center
    if first_gh_idx is not None and first_gh_idx > 0:
        prev = mol_lines[first_gh_idx - 1]
        if prev.upper().startswith("H"):
            parts = prev.split()
            gx.insert(0, float(parts[1]))
            gy.insert(0, float(parts[2]))

    return np.array(gx), np.array(gy)

file1 = "ohba_fpb.out"
# === Parse data ===
fpb_x, fpb_y = parse_proton_traj(
    file1
)
ctpb_x, ctpb_y = parse_proton_traj(
    "ohba_ctpb.out"
)
gh_x, gh_y = parse_basis_centers(
    file1
)

print(f"Basis centers ({len(gh_x)}):")
for x, y in zip(gh_x, gh_y):
    print(f"  ({x:.4f}, {y:.4f})")

# Timestep in fs — adjust to match your simulation
dt_fs = 0.0242*0.04
idx_18fs = 19110 #int(18.0 / dt_fs)
idx_18fsa = int(idx_18fs/10)
print(19110, idx_18fs, 19110*0.0242*0.04)

# === Plot ===
fig, ax = plt.subplots(figsize=(9, 8))

# c-TPB trajectory (30 000 points — subsample for rendering speed without loss of coverage)
subsample = 3
ax.scatter(
    ctpb_x[::subsample], ctpb_y[::subsample],
    s=8, alpha=0.25, color="blue", rasterized=True, zorder=1
)

# FPB trajectory
ax.scatter(
    fpb_x, fpb_y,
    s=8, alpha=0.5, color="red", rasterized=True, zorder=2
)

# Square markers at 18 fs
ax.scatter(
    ctpb_x[idx_18fs], ctpb_y[idx_18fs],
    s=250, marker="s", color="blue", edgecolors="none", linewidths=1.5, zorder=7
)
ax.scatter(
    fpb_x[idx_18fsa], fpb_y[idx_18fsa],
    s=250, marker="s", color="red", edgecolors="none", linewidths=1.5, zorder=7
)

# Trajectory start markers (unlabeled)
ax.scatter(
    ctpb_x[0], ctpb_y[0],
    s=200, marker="o", color="blue", edgecolors="none", linewidths=1.5, zorder=6
)
ax.scatter(
    fpb_x[0], fpb_y[0],
    s=200, marker="o", color="red", edgecolors="none", linewidths=1.5, zorder=6
)

ccolor = "#55BB55"
# Fixed proton basis function centers
ax.scatter(
    gh_x, gh_y,
    s=300, marker="*", color=ccolor, zorder=15
)

# === Formatting ===
ax.set_xlabel(r"$x$ (Å)", fontsize=36)
ax.set_ylabel(r"$y$ (Å)", fontsize=36)
ax.tick_params(axis="both", which="major", length=10, width=4, labelsize=28)

for spine in ax.spines.values():
    spine.set_linewidth(4)
    spine.set_color("black")

legend_handles = [
    Line2D([0, 1], [0, 0], color="blue", linewidth=3, alpha=1, label="c-TPB"),
    Line2D([0, 1], [0, 0], color="red", linewidth=3, alpha=1, label="FPB"),
    Line2D([0], [0], marker="*", color=ccolor, markersize=16,
           linestyle="None", label="Basis centers"),
]
legend = ax.legend(handles=legend_handles, fontsize=24, loc="upper right")

# Molecule image inset (lower-left, white margins cropped)
mol_img = mpimg.imread(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "molecule.png")
)
h, w = mol_img.shape[:2]
mol_img = mol_img[int(h * 0.03):int(h * 0.97), int(w * 0.03):int(w * 0.97)]
axins = ax.inset_axes([0.01, 0.01, 0.25, 0.25])
axins.imshow(mol_img)
axins.axis("off")

plt.tight_layout()
output_file = "FigS2.png"
plt.savefig(output_file, dpi=300)
print(f"Saved {output_file}")
plt.show()
