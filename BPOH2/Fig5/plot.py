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


filenames = ["nick.txt"]
names = ["c-TPB"]

# Electronic timestep in a.u. for the c-TPB BPOH2 simulation
dt = 0.1

plt.figure(figsize=(11, 8))

for i, (fname, label) in enumerate(zip(filenames, names)):
    data = load_series(fname)
    if data is None:
        continue

    t = np.arange(data.size) * dt
    data, t = data[::10], t[::10]

    plt.plot(t * 2.418884e-2, 1e5 * (data - data[0]),
             linestyle='-', label=label, lw=4, color="blue")

plt.xlabel("Time (fs)", fontsize=36)
plt.ylabel(r"Energy Change ($10^{-5}$ a.u.)", fontsize=36)
plt.xlim(0, 18)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.tick_params(axis='both', which='major', length=10, width=4, colors='black')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
    spine.set_color("black")

plt.tight_layout()
plt.savefig("Fig5.png", dpi=300)
plt.show()
