import matplotlib.pyplot as plt
import numpy as np
import os
import re
from glob import glob

# === Settings ===
scale = 0.5
box_width = 0.03  # Same as ±0.015 in Gnuplot
output_filename = "aniso_del.png"

# === Read all files matching pattern ===
file_pattern = "nfb.imag_aniso_gap0_*.00"
file_list = sorted(glob(file_pattern))

temp_file_pairs = []
temp_pattern = re.compile(r"nfb\.imag_aniso_gap0_(\d+\.\d+)$")
for f in file_list:
    match = temp_pattern.search(f)
    if match:
        temp = float(match.group(1))
        temp_file_pairs.append((temp, f))

# Sort by temperature
temp_file_pairs.sort(key=lambda x: x[0])
temperatures = [pair[0] for pair in temp_file_pairs]
num_temps = len(temperatures)

# === Setup colormap for colors ===
colormap = plt.colormaps['rainbow'].resampled(num_temps)
temp_to_color = {temp: colormap(i) for i, temp in enumerate(temperatures)}

# === Determine x-range from temps ===
xrange = (min(temperatures) - 1, max(temperatures) + 1)
yrange = (0, 7)

# === Initialize plot ===
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(xrange)
ax.set_ylim(yrange)

ax.set_xlabel("Temperature (K)", fontsize=25)
ax.set_ylabel(r"$\Delta_{nk}$ (meV)", fontsize=25)
ax.tick_params(axis='both', labelsize=20)

ax.set_xticks(temperatures)
ax.set_xticklabels([f"{t:.1f}" if t % 1 else f"{int(t)}" for t in temperatures])
ax.set_yticks(np.arange(0, 7, 0.5))

# === Vertical lines ===
for t in range(int(np.floor(xrange[0])), int(np.ceil(xrange[1])) + 1):
    ax.axvline(x=t, color='gray', linestyle='--', linewidth=0.5, zorder=0)

# === Draw histograms with rainbow colors ===
for temp, filename in temp_file_pairs:
    try:
        data = np.loadtxt(filename, comments='#')
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    gap_vals = data[:, 0]  # x in Gnuplot
    densities = data[:, 1]  # y in Gnuplot

    # Map to plot coords
    x_centers = ((gap_vals - temp) * scale / 2) + temp
    x_widths = (gap_vals - temp) * scale

    color = temp_to_color[temp]

    for x, y, dx in zip(x_centers, densities, x_widths):
        # Draw horizontal box for each point with temperature color
        rect = plt.Rectangle((x - dx / 2, y - box_width / 2),
                             width=dx,
                             height=box_width,
                             color=color)
        ax.add_patch(rect)

# === Finalize and save ===
plt.tight_layout()
plt.savefig(output_filename, dpi=150)
plt.close()
print(f"Plot saved as: {output_filename}")

