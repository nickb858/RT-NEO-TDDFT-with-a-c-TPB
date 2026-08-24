import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

ANGSTROM_TO_BOHR = 1.8897259886


def parse_xyz_blocks(filename):
    """Parse a Q-Chem NEO-TDDFT output file.

    Returns a list of geometry frames. Each frame is a list of (atom, x, y, z)
    tuples in Angstrom, with the quantum proton expectation position (from
    [N] Dipole Moment, converted from a.u. to Angstrom) appended as the
    last entry.
    """
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
                    blocks[-1].append(("H",
                                       float(x) / ANGSTROM_TO_BOHR,
                                       float(y) / ANGSTROM_TO_BOHR,
                                       float(z) / ANGSTROM_TO_BOHR))
    return blocks


def distance(block, idx1, idx2):
    """Return the distance (Å) between two atoms in a frame (1-based indices)."""
    a = np.array(block[idx1 - 1][1:])
    b = np.array(block[idx2 - 1][1:])
    return np.linalg.norm(a - b)


# Files and method labels (must be run from this directory)
files = ["ohba_tpb.out", "ohba_otpb.out", "ohba_fpb.out"]
names = ["c-TPB", "Original TPB", "FPB"]

# Timestep in fs: nuclear update every 10 electronic steps × dt=0.04 a.u.
dt_base = 0.0096755

plt.figure(figsize=(11, 8))
linestyles = ["-", "--", ":", (0, (3, 1, 1, 1))]

for i, (file, label) in enumerate(zip(files, names)):
    dt = dt_base
    if file == "ohba_fpb.out":
        dt *= 10  # FPB uses DT=0.4 a.u. vs 0.04 a.u. for c-TPB/Original TPB

    blocks = parse_xyz_blocks(file)[:-1][:3000]

    # 1-based index of the appended quantum proton position in each frame.
    # FPB has 3 additional ghost atoms, pushing the proton to index 19.
    proton_idx = 16 if i < 2 else 19

    d_donor    = [distance(block, 7, proton_idx) for block in blocks]
    d_acceptor = [distance(block, proton_idx, 9) for block in blocks]
    times = np.arange(len(blocks)) * dt

    ls = linestyles[i]
    plt.plot(times, d_donor,    lw=4, color="blue", linestyle=ls, label=label)
    plt.plot(times, d_acceptor, lw=4, color="red",  linestyle=ls)

plt.text(0.02, 0.26, r"O$_{\rm D}$—H", color="blue", fontsize=36,
         transform=plt.gca().transAxes)
plt.text(0.02, 0.80, r"O$_{\rm A}$—H", color="red",  fontsize=36,
         transform=plt.gca().transAxes)

plt.ylabel("Distance (Å)", fontsize=40)
plt.xlabel("Time (fs)", fontsize=40)
plt.xlim(0, dt_base * 3000)
plt.ylim(0.9, 2.1)

leg = plt.legend(fontsize=36, loc="upper right")
for line in leg.get_lines():
    line.set_color('black')

plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.tick_params(axis='both', which='major', length=10, width=4, colors='black')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
    spine.set_color("black")

plt.tight_layout()
plt.savefig("Fig2.png", dpi=300)
plt.show()
