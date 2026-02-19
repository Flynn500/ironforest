import time
import numpy as np
import matplotlib.pyplot as plt
import ironforest as irn
from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs


def make_grid(resolution=200, extent=(0.0, 1.0)):
    """Create a 2D grid of query points."""
    lo, hi = extent
    x = np.linspace(lo, hi, resolution)
    y = np.linspace(lo, hi, resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def run_kde(points_np, grid_np, bandwidth=0.05, leaf_size=32):
    """Run KDE with all 3 methods, return density grids and timings."""
    points_irn = irn.Array.from_numpy(points_np)
    grid_irn = irn.Array.from_numpy(grid_np)

    # --- AggTree ---
    t0 = time.perf_counter()
    agg_density = irn.spatial.AggTree.from_array(
        points_irn, leaf_size=leaf_size, metric="euclidean", kernel="gaussian", bandwidth=bandwidth, atol=0.25,
    ).kernel_density(grid_irn, bandwidth=bandwidth, kernel="gaussian", normalize=True)
    agg_time = time.perf_counter() - t0
    agg = np.array(irn.Array.to_numpy(agg_density)) # type: ignore

    # --- irn KDTree ---
    t0 = time.perf_counter()
    kd_density = irn.spatial.KDTree.from_array(
        points_irn, leaf_size=leaf_size, metric="euclidean"
    ).kernel_density(grid_irn, bandwidth=bandwidth, kernel="gaussian", normalize=True)
    kd_time = time.perf_counter() - t0
    kd = np.array(irn.Array.to_numpy(kd_density)) # type: ignore

    # --- sklearn KDTree ---
    t0 = time.perf_counter()
    sk_density = KDTree(
        points_np, leaf_size=leaf_size, metric="euclidean"
    ).kernel_density(grid_np, h=bandwidth, kernel="gaussian")
    sk_time = time.perf_counter() - t0
    sk = np.array(sk_density)

    return (agg, kd, sk), (agg_time, kd_time, sk_time)


def plot_heatmaps(xx, yy, densities, times, output_path="kde_heatmap_compare.png"):
    """Plot 3 KDE heatmaps side-by-side with timing info."""
    labels = ["irn AggTree", "irn KDTree", "sklearn KDTree"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    resolution = xx.shape[0]
    vmin = min(d.min() for d in densities)
    vmax = max(d.max() for d in densities)

    for ax, density, label, t in zip(axes, densities, labels, times):
        im = ax.pcolormesh(
            xx, yy, density.reshape(resolution, resolution),
            shading="auto", cmap="inferno", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{label}\n({t:.3f}s)", fontsize=13)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes, label="Density", shrink=0.85)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Generate clustered 2D data
    points_np, _ = make_blobs( # type: ignore
        n_samples=50_000, centers=10, cluster_std=0.04,
        n_features=2, random_state=101,
    )
    # Rescale to [0, 1]
    points_np -= points_np.min(axis=0)
    points_np /= points_np.max(axis=0)

    xx, yy, grid_np = make_grid(resolution=200)
    densities, times = run_kde(points_np, grid_np, bandwidth=0.1)
    plot_heatmaps(xx, yy, densities, times)