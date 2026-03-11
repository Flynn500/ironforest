"""Compare spatial (Gaussian projection) vs standard splits for Random Forest.

Classification : 2-D sin-boundary dataset  →  decision boundary plots
Regression     : 1-D noisy sin curve       →  fitted curve overlay plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from ironforest.models import RandomForestClassifier, RandomForestRegressor


# ── Datasets ──────────────────────────────────────────────────────────────────

def make_clf_dataset(n=500, noise=0.10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, (n, 2))
    y = (np.sin(X[:, 0]) > X[:, 1] + rng.normal(0, noise, n)).astype(int)
    return X, y


def make_reg_dataset(n=200, noise=0.25, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, noise, n)
    return X, y


# ── Plot helpers ──────────────────────────────────────────────────────────────

CMAP_BG  = ListedColormap(["#aec6f5", "#f5aeae"])
CMAP_PTS = ListedColormap(["#1a6eff50", "#ff222250"])


def plot_clf_boundary(ax, model, X, y, title):
    h = 0.05
    x0_min, x0_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    x1_min, x1_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h),
                         np.arange(x1_min, x1_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=CMAP_BG)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_PTS,
               edgecolors="k", linewidths=0.0, s=5, zorder=3)
    xs = np.linspace(x0_min, x0_max, 300)
    ax.plot(xs, np.sin(xs), "k--", lw=1.4, label="true boundary")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
    ax.legend(fontsize=7, loc="upper right")


def plot_reg_curve(ax, model, X, y, title):
    xs = np.linspace(X.min() - 0.1, X.max() + 0.1, 400).reshape(-1, 1)
    ys_pred = model.predict(xs)
    ys_true = np.sin(xs[:, 0])

    ax.scatter(X[:, 0], y, s=14, color="#888", alpha=0.5, zorder=2, label="data")
    ax.plot(xs[:, 0], ys_true, "k--", lw=1.4, label="true sin(x)")
    ax.plot(xs[:, 0], ys_pred, color="#e05c00", lw=2.0, label="RF prediction")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7, loc="upper right")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_clf, y_clf = make_clf_dataset()
    X_reg, y_reg = make_reg_dataset()

    n_estimators = 100
    depths = [3]

    fig = plt.figure(figsize=(15, 12))
    # fig.suptitle(
    #     "Random Forest: Standard splits  vs  Gaussian projection splits",
    #     fontsize=13, fontweight="bold", y=0.99,
    # )

    # 4 rows: clf-standard, clf-spatial, reg-standard, reg-spatial
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.30)

    row_meta = [
        # ("Classification — Standard (gini)", "clf", "gini"),
        # ("Classification — RandomProjection (gaussian)", "clf", "random_projection"),
        ("Regression — Standard (mse)", "reg", "mse"),
        ("Regression — RandomProjection (gaussian)", "reg", "random_projection"),
    ]

    for row, (row_label, task, criterion) in enumerate(row_meta):
        # fig.text(0.01, 0.88 - row * 0.235, row_label,
        #          va="center", rotation="vertical", fontsize=9, color="#333")

        for col, depth in enumerate(depths):
            depth_str = f"depth={depth}" if depth else "depth=∞"
            ax = fig.add_subplot(gs[row, col])

            if task == "clf":
                mdl = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    criterion=criterion, # type: ignore
                    projection_type="gaussian",
                    projection_density=0.5,
                    random_state=0,
                )
                mdl.fit(X_clf, y_clf)
                plot_clf_boundary(ax, mdl, X_clf, y_clf,
                                  f"{criterion} | {depth_str} | n={n_estimators}")
            else:
                mdl = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    criterion=criterion, # type: ignore
                    projection_type="gaussian",
                    projection_density=0.5,
                    random_state=0,
                )
                mdl.fit(X_reg, y_reg)
                plot_reg_curve(ax, mdl, X_reg, y_reg,
                               f"{criterion} | {depth_str} | n={n_estimators}")

    plt.savefig("rf_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved → rf_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()