import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# Collect all matching files
file_list = sorted(glob.glob("ncn.qdos_*.00"))

plt.figure(figsize=(8, 6))

for file in file_list:
    # Extract temperature from filename using regex
    temp_match = re.search(r"qdos_(\d+\.\d+)", file)
    if not temp_match:
        continue
    temperature = float(temp_match.group(1))

    # Load data
    data = np.loadtxt(file, skiprows = 1)
    w = data[:, 0]  # energy (eV)
    dos = data[:, 1]  # N_S/N_F

    # Create mirrored energy and DOS
    w_full = np.concatenate((-w[::-1], w))
    dos_full = np.concatenate((dos[::-1], dos))

    # Plot
    plt.plot(w_full * 1e3, dos_full, label=f"{temperature:.2f} K")  # Convert eV to meV

plt.xlim(-5,5)
plt.xlabel(r"$\omega$ [meV]", fontsize=14)
plt.ylabel(r"$N_S/N_F$", fontsize=14)
plt.legend(title="Temperature", fontsize=10)
plt.grid(True)
plt.tight_layout()
#plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.savefig("qdos_mirrored_vs_temperature.png")
plt.show()

