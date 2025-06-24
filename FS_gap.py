import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.collections import LineCollection
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.spatial.transform import Rotation as R
import os

# Matplotlib global settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 24,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_gap_FS(filename):
    records = []
    with open(filename) as f:
        for line in f:
            if line.strip() == "" or line.lstrip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                kx, ky, kz = map(float, parts[0:3])
                band = int(parts[3])
                enk_ef = float(parts[4])
                gap = float(parts[5])
                records.append([kx, ky, kz, band, enk_ef, gap])
            except (ValueError, IndexError):
                continue
    if not records:
        raise ValueError("No valid data found in the file")
    data = np.array(records)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def get_available_bands(data):
    return np.unique(data[:, 3].astype(int))

def get_band_filename(bands, theta, phi, kz_tol):
    bands_str = "_".join(str(b) for b in sorted(bands))
    return f"gap_FS_bands_{bands_str}_theta{theta}_phi{phi}_kztol{kz_tol}.png"

def plot_slanted_plane(data, selected_bands, filename, theta_degrees=0, phi_degrees=0, kz_tolerance=0.05):
    fig, ax = plt.subplots(figsize=(13, 10), layout='constrained', facecolor='black')
    ax.set_facecolor('black')
    ngrid = 500

    # Rotation
    theta = np.radians(theta_degrees)
    phi = np.radians(phi_degrees)
    rot = R.from_euler('zy', [phi, theta])

    # High-symmetry points
    high_sym_points = [
        (0.0000000000, 0.0000000000, 0.0000000000, "Γ"),
        (0.5000000000, 0.5000000000, 0.5000000000, "T"),
        (0.8125178462, 0.1874821538, 0.5000000000, "H₂"),
        (0.5000000000, -0.1874821538, 0.1874821538, "H₀"),
        (0.5000000000, 0.0000000000, 0.0000000000, "L"),
        (0.0000000000, 0.0000000000, 0.0000000000, "Γ")
    ]

    k_points = data[:, :3]
    rotated_k = rot.apply(k_points)
    rotated_data = np.column_stack((rotated_k, data[:, 3:]))

    all_gaps = []
    for b in selected_bands:
        mask = (data[:, 3] == b) & (np.abs(rotated_data[:, 2]) < kz_tolerance)
        all_gaps.extend(data[mask, 5])
    if not all_gaps:
        raise ValueError("No data points found within ±{} of slanted plane".format(kz_tolerance))

    vmin, vmax = np.nanmin(all_gaps), np.nanmax(all_gaps)
    cmap = mpl.colormaps.get('gist_heat')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for idx, b in enumerate(selected_bands):
        mask = (data[:, 3] == b) & (np.abs(rotated_data[:, 2]) < kz_tolerance)
        d = data[mask]
        if len(d) == 0:
            continue
        kx_rot = rotated_data[mask, 0]
        ky_rot = rotated_data[mask, 1]
        enk_ef = d[:, 4]
        gap = d[:, 5]

        xi = np.linspace(np.min(kx_rot), np.max(kx_rot), ngrid)
        yi = np.linspace(np.min(ky_rot), np.max(ky_rot), ngrid)
        xi, yi = np.meshgrid(xi, yi)
        zi_ene = griddata((kx_rot, ky_rot), enk_ef, (xi, yi), method='cubic')

        cs = ax.contour(xi, yi, zi_ene, levels=[0], colors='none')
        fs_points = []
        fs_gap = []

        for collection in cs.collections:
            for path in collection.get_paths():
                vertices = path.vertices
                px, py = vertices[:, 0], vertices[:, 1]
                p_gap = griddata((kx_rot, ky_rot), gap, (px, py), method='cubic')
                fs_points.append(np.c_[px, py])
                fs_gap.append(p_gap)

        for seg, lam in zip(fs_points, fs_gap):
            if len(seg) < 2:
                continue
            points_ = seg.reshape(-1, 1, 2)
            segments = np.concatenate([points_[:-1], points_[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=cmap, norm=norm,
                linewidth=3.5,
                zorder=10 + idx
            )
            lc.set_array(lam)
            ax.add_collection(lc)

    # Plot high-symmetry points
    for point in high_sym_points:
        kx, ky, kz, label = point
        rotated_point = rot.apply([kx, ky, kz])
        if np.abs(rotated_point[2]) < kz_tolerance:
            ax.plot(rotated_point[0], rotated_point[1], 'wo', markersize=7, zorder=20)
            offset = 0.03 * max(np.ptp(rotated_data[:, 0]), np.ptp(rotated_data[:, 1]))
            ax.text(rotated_point[0] + offset, rotated_point[1] + offset, label,
                    fontsize=20, ha='center', va='center', zorder=21,
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    # Inset colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax.inset_axes([1.02, 0.0, 0.035, 1.0])  # [x0, y0, width, height] relative to ax
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Δ(0) (meV)', rotation=270, labelpad=25, fontsize=24, color='white')
    cbar.ax.tick_params(labelsize=20, width=1.5, length=5, colors='white')

    # Axis styling
    ax.set_xlabel(r'$K_x$', labelpad=10, color='white')
    ax.set_ylabel(r'$K_y$', labelpad=10, color='white')
    #ax.set_aspect('equal')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color('white')

    kx_rot_all = rotated_data[:, 0]
    ky_rot_all = rotated_data[:, 1]
    kx_min, kx_max = np.min(kx_rot_all), np.max(kx_rot_all)
    ky_min, ky_max = np.min(ky_rot_all), np.max(ky_rot_all)

    k_range = max(kx_max - kx_min, ky_max - ky_min)
    major_tick = np.round(k_range / 4, 2)
    minor_tick = major_tick / 5

    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.yaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick))

    padding = 0.05
    ax.set_xlim(-2,2)
    ax.set_ylim(-0.25,1.25)
    #ax.set_xlim(kx_min - padding, kx_max + padding)
    #ax.set_ylim(ky_min - padding, ky_max + padding)

    plt.savefig(filename, dpi=600)
    plt.close()
    print(f"✓ Saved: {filename}")

def main():
    filename = "ncn.imag_aniso_gap_FS_003.00"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file '{filename}' not found")

    try:
        data = load_gap_FS(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    bands_available = get_available_bands(data)
    print("Available bands in data:", bands_available)

    sel_bands = input("Enter band numbers to plot (space/comma separated, default all): ").replace(',', ' ').split()
    selected_bands = [int(b) for b in sel_bands if b.isdigit()] if sel_bands else bands_available

    if not selected_bands:
        print("No valid bands selected")
        return

    theta_degrees = float(input("Enter polar rotation angle θ (degrees, default 0): ") or 0)
    phi_degrees = float(input("Enter azimuthal rotation angle φ (degrees, default 0): ") or 0)
    kz_tol = float(input("Enter kz tolerance (default 0.05): ") or 0.05)

    print(f"Selected bands: {selected_bands}")
    print(f"Plane rotation: θ={theta_degrees}°, φ={phi_degrees}°")
    print(f"kz tolerance: ±{kz_tol}")

    outfilename = get_band_filename(selected_bands, theta_degrees, phi_degrees, kz_tol)

    try:
        plot_slanted_plane(data, selected_bands, outfilename,
                           theta_degrees=theta_degrees,
                           phi_degrees=phi_degrees,
                           kz_tolerance=kz_tol)
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    main()

