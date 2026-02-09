from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import time

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def plot_live(x, f, x_, xxx_, objectives):

    # give a binary mask of feasiblity, so if feas==1, then it is feasible.
    feas = np.array([int(fi[0] < 1e-3) for fi in f])

    # Stack real and sample points together
    x_xxx_ = np.vstack([x_, xxx_])

    # Reduce to 2D latent space
    start = time.time()
    x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)

    # Split back to x and xxx projections
    x_len = x.shape[0]
    if x_len < 30:
        raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")
    x__1 = x_xxx_latent[:x_len, :]
    x__ = TSNE(n_components=2).fit_transform(x__1)
    xxx__ = x_xxx_latent[x_len:, :]
    x_len = x.shape[0]
    x__ = x_xxx_latent[:x_len, :]
    end = time.time()
    print(
        f'\nFAST latent embedding took: {end - start}s\nFor a more structured plot please use TSNE on x__1=PCA(x_xxx_) with Mahalanobis distance')

    # Train KNN on the latent projection
    k = 1
    f_hat_latent = KNeighborsClassifier(n_neighbors=k)
    f_hat_latent.fit(x__, f.ravel())
    f_hat_pred_latent = f_hat_latent.predict(xxx__)
    f_hat_pred_latent = f_hat_pred_latent.reshape(-1, 1)

    # Estimate resolution
    res = int(np.sqrt(xxx__.shape[0]))

    # === Plotting ===
    fig, ax = plt.subplots()

    # Colours based on objective value.
    # take the first objective value always !!
    f1 = np.array([obj[0] for obj in objectives])
    colors = cm.viridis(f1 / np.max(f1))  # scale to [0, 1]

    # Plot each actual x point
    for xi, yi, ci, fi in zip(x__[:, 0], x__[:, 1], colors, feas):
        ax.scatter(
            xi, yi,
            color=ci,
            edgecolors='black' if fi == 1 else 'none',
            linewidths=1,
            s=60,
            zorder=2,
            rasterized=True,
        )

    # Shade KNN-predicted boundary in latent space
    for i in range(xxx__.shape[0]):
        if f_hat_pred_latent[i, 0] == 0:
            ax.scatter(
                xxx__[i, 0], xxx__[i, 1],
                color='green',
                alpha=0.3,
                s=25,
                edgecolors='none',
                zorder=1,
                rasterized=True
            )
        else:
            ax.scatter(
                xxx__[i, 0], xxx__[i, 1],
                color='grey',
                alpha=0.1,
                s=25,
                edgecolors='none',
                zorder=1,
                rasterized=True
            )

    # Colorbar
    norm = mcolors.Normalize(vmin=np.min(f1), vmax=np.max(f1))
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Apogee (ft)")

    ax.set_xlabel("Latent x1")
    ax.set_ylabel("Latent x2")
    ax.set_title("Latent Design Space with KNN Boundary")
    ax.grid(True, zorder=1)

    plt.savefig("live_plot.png", dpi=150)
    print('latent plot generated')
    plt.close()

def plot_latent(x, f, x_, xxx_, objectives):

    # give a binary mask of feasiblity, so if feas==1, then it is feasible.
    feas = np.array([int(fi[0] < 1e-3) for fi in f])

    # Stack real and sample points together
    x_xxx_ = np.vstack([x_, xxx_])

    # Reduce to 2D latent space
    start = time.time()
    x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)

    # Split back to x and xxx projections
    x_len = x.shape[0]
    if x_len < 30:
        raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")
    x__1 = x_xxx_latent[:x_len, :]
    x__ = TSNE(n_components=2).fit_transform(x__1)
    # from MulticoreTSNE import MulticoreTSNE as TSNE
    # tsne = TSNE(n_jobs=4) # n_jobs is number of cores (4=1.2x speedup)
    # x__ = tsne(n_components=2).fit_transform(x__1)
    end = time.time()
    print(f'\nFAST latent embedding took: {end - start}s\nFor a more structured plot please use TSNE on x__1=PCA(x_xxx_) with Mahalanobis distance')

    # === Plotting ===
    fig, ax = plt.subplots()

    # Colours based on objective value.
    # take the first objective value always !!
    f1 = np.array([obj[0] for obj in objectives])
    colors = cm.viridis(f1 / np.max(f1))  # scale to [0, 1]

    # Plot each actual x point
    for xi, yi, ci, fi in zip(x__[:, 0], x__[:, 1], colors, feas):
        ax.scatter(
            xi, yi,
            color=ci,
            edgecolors='black' if fi == 1 else 'none',
            linewidths=1,
            s=60,
            zorder=2
        )

    # Colorbar
    norm = mcolors.Normalize(vmin=np.min(f1), vmax=np.max(f1))
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Objective: f1")

    ax.set_xlabel("Latent x1")
    ax.set_ylabel("Latent x2")
    ax.set_title("Latent Design Space with Feasbile Outlined in Black")
    ax.grid(True, zorder=1)

    plt.savefig("latent_plot.png")
    print('Saved latent_plot.png')
    plt.close()

