import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

ANGSTROM_TO_BOHR = 1.8897259886


def parse_xyz_blocks(filename):
    """Parse a Q-Chem NEO-TDDFT output file (nuclear geometry in Bohr).

    Returns a list of geometry frames. Each frame is a list of (atom, x, y, z)
    tuples in Angstrom (converted from Bohr), followed by the expectation
    positions of quantum proton 0 and proton 1 (also converted from Bohr).
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
                    current.append((atom,
                                    float(x) / ANGSTROM_TO_BOHR,
                                    float(y) / ANGSTROM_TO_BOHR,
                                    float(z) / ANGSTROM_TO_BOHR))
            elif in_block and not line.strip():
                if current:
                    blocks.append(current)
                in_block = False
                aft = True
            elif " Expectation value of proton 0 :" in line:
                if aft:
                    _, _, _, _, _, _, x, y, z = line.split()
                    blocks[-1].append(("H",
                                       float(x) / ANGSTROM_TO_BOHR,
                                       float(y) / ANGSTROM_TO_BOHR,
                                       float(z) / ANGSTROM_TO_BOHR))
            elif " Expectation value of proton 1 :" in line:
                if aft:
                    aft = False
                    _, _, _, _, _, _, x, y, z = line.split()
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


files = ["BPOH2_TPB5.out"]
names = ["c-TPB"]

# Timestep in fs: nuclear update every 10 electronic steps × dt=0.1 a.u.
dt = 0.0242 * 10 * 0.1

# 1-based index of quantum proton 1 (the transferring proton):
# 22 classical atoms + proton 0 + proton 1 = index 24
proton_idx = 24

plt.figure(figsize=(11, 8))

for i, (file, label) in enumerate(zip(files, names)):
    blocks = parse_xyz_blocks(file)[:-1][:3000]
    times = np.arange(len(blocks)) * dt

    # Atom 11: O donor; atom 12: N acceptor; atom 24: quantum proton 1
    d_donor    = [distance(block, 11, proton_idx) for block in blocks]
    d_acceptor = [distance(block, 12, proton_idx) for block in blocks]

    plt.plot(times, d_donor,    lw=4, color="blue", label=f"Donor {label}")
    plt.plot(times, d_acceptor, lw=4, color="red",  label=f"Acceptor {label}")

plt.text(0.02, 0.06, r"O$_{\rm D}$—H", color="blue", fontsize=36,
         transform=plt.gca().transAxes)
plt.text(0.02, 0.78, r"N$_{\rm A}$—H", color="red",  fontsize=36,
         transform=plt.gca().transAxes)

plt.ylabel("Distance (Å)", fontsize=36)
plt.xlabel("Time (fs)", fontsize=36)
plt.xlim(0, 18)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.tick_params(axis='both', which='major', length=10, width=4, colors='black')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
    spine.set_color("black")

plt.tight_layout()
plt.savefig("Fig4.png", dpi=300)
plt.show()
