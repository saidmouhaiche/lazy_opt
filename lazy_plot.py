from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import time

# from sklearn.manifold import TSNE
from sklearn.manifold import TSNE as sklearnTSNE
# from tsnecuda import TSNE
from openTSNE.sklearn import TSNE as openTSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def latent_pca(x_, xxx_):
    # Stack real and sample points together
    x_xxx_ = np.vstack([x_, xxx_])
    x_len = x_.shape[0]

    x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)

    xxx__ = x_xxx_latent[x_len:, :]
    x__ = x_xxx_latent[:x_len, :]

    return x__, xxx__

def latent_tsne(x_):
    # Split back to x and xxx projections
    x_len = x_.shape[0]
    if x_len < 30:
        raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")

    # CUDA GPU accelerated tsne
    x__ = openTSNE(n_components=2,n_jobs=-1).fit_transform(x_)

    # tsne is too slow to return xxx__
    return x__

def latent_tsne_3d(x_):
    # Split back to x and xxx projections
    x_len = x_.shape[0]
    if x_len < 30:
        raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")

    x__ = sklearnTSNE(n_components=3, n_jobs=-1).fit_transform(x_)

    return x__


def latent_knn(x__,xxx__,f):
    # Train KNN on the latent projection
    k = 1
    f_hat_latent = KNeighborsClassifier(n_neighbors=k)
    f_hat_latent.fit(x__, f.ravel())
    f_hat_pred_latent = f_hat_latent.predict(xxx__)
    f_hat_pred_latent = f_hat_pred_latent.reshape(-1, 1)

    return f_hat_pred_latent

def latent_pca_tsne(x_,xxx_):
    x__pca, xxx__pca = latent_pca(x_, xxx_)
    x__ = latent_tsne(x__pca)
    return x__

def latent_classifier(lazy, x_, xxx_, f):
    surr = lazy.hyper_params[0]
    f_hat = lazy._create_surrogate(x_, f, surr)
    f_hat_pred_latent = f_hat.predict(np.array(xxx_, dtype=np.float32))
    f_hat_pred_latent = f_hat_pred_latent.reshape(-1, 1)
    return f_hat_pred_latent



def plot_langevitour(x, feasible, bounds=None, output_file="optimization_tour.html"):
    """
    Create an interactive langevitour visualization of the optimization process.

    Parameters:
    -----------
    x : np.ndarray
        Design variables (N, dims) - the actual search points
    f : np.ndarray
        Feasibility values (N, 1) - binary feasibility
    objectives : list of tuples
        List of objective values for each point, e.g., [(f1,), (f2,), ...]
    feasible : list of bool
        Feasibility status for each point
    bounds : list, optional
        Bounds for each dimension [lower1, upper1, lower2, upper2, ...]
    output_file : str
        Output HTML filename

    Returns:
    --------
    tour : Langevitour object
        The tour visualization object
    """
    try:
        from langevitour import Langevitour
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: langevitour is not installed!")
        print("Install it with: pip install langevitour")
        print("="*60 + "\n")
        return None

    # Convert data to proper format
    X = x
    n_points = X.shape[0]
    dims = X.shape[1]

    # Create axis labels
    if bounds is not None:
        axis_labels = []
        for i in range(dims):
            lower = bounds[2*i]
            upper = bounds[2*i+1]
            axis_labels.append(f"x{i+1} [{lower:.2f}, {upper:.2f}]")
    else:
        axis_labels = [f"x{i+1}" for i in range(dims)]

    # Create the tour
    print("\nCreating langevitour visualization...")
    print(f"  Points: {n_points}")
    print(f"  Dimensions: {dims}")
    # print(f"  Groups: {len(set(groups))}")
    print(f"  Feasible points: {sum(feasible)}")

    tour = Langevitour(
        data=X,
        # group=groups,
        column_names=axis_labels,
        point_size=2.5
    )

    # Save to HTML
    tour.write_html(output_file)
    print(f"\n✓ Interactive visualization saved to: {output_file}")
    print(f"  Open this file in a web browser to explore your optimization space!")
    print(f"\n  Features:")
    print(f"    - Drag to rotate the projection")
    print(f"    - Use GUI to show/hide groups")
    print(f"    - Enable 'Guide' mode for projection pursuit")
    print(f"    - Select specific axes to focus on")

    return tour


def plot_langevitour_with_hex_model(
        high_d_data,
        layout_2d,
        groups=None,
        num_bins=None,
        filter_edges=True,
        edge_cutoff=None,
        column_names=None,
        output_file="optimization_tour.html"):
    """
    Langevitour visualization with the paper's hexagonal bin wireframe model.

    Shows both the original high-D data points and hex bin centroids in
    high-D space, connected by Delaunay triangulation edges.  As the tour
    rotates you can see where the 2D layout faithfully represents high-D
    structure (short edges stay short) and where it distorts it.

    Parameters
    ----------
    high_d_data : np.ndarray, shape (n, p)
        Original high-dimensional data points (e.g. normalised design vars).
    layout_2d : np.ndarray, shape (n, 2)
        2D embedding used to compute hexagonal bins (e.g. PCA of x_).
    groups : list[str] or np.ndarray, optional
        Group label for each of the n data points.  Defaults to "data".
        Useful values: "feasible" / "infeasible" for optimisation problems.
    num_bins : int, optional
        Hexbin gridsize.  Auto-computed from Freedman-Diaconis rule if None.
    filter_edges : bool
        Remove long triangulation edges using automatic gap criterion.
    edge_cutoff : float, optional
        Manual 2-D distance cutoff for edge filtering (overrides automatic).
    column_names : list[str], optional
        Axis labels for the tour dimensions.
    output_file : str
        Output HTML filename.

    Returns
    -------
    tour : Langevitour object or None
    """
    try:
        from langevitour import Langevitour
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: langevitour is not installed!")
        print("Install it with: pip install langevitour")
        print("="*60 + "\n")
        return None


    """
    Hexagonal binning + Delaunay triangulation pipeline.

    Converted from R (hexbin + tripack packages) to Python.
    Used to build a wireframe 'model' of the data structure in high-D space,
    overlaid on a Langevitour grand tour plot — as described in the paper:
      "Choosing better NLDR layouts by evaluating the model in the
       high-dimensional data space" (Gamage et al., 2025)

    Pipeline:
      2D layout (n x 2)
          -> hexagonal bins
          -> bin centroids in 2D  (m x 2)
          -> bin centroids in high-D  (m x p)  [mean of points per bin]
          -> Delaunay triangulation of 2D centroids
          -> edge pairs (line_from, line_to)  [0-based into the m centroids]
          -> Langevitour( [centroids | data], group, line_from, line_to )
    """

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay, cKDTree
    from scipy.stats import iqr

    # ---------------------------------------------------------------------------
    # Bin width
    # ---------------------------------------------------------------------------

    def find_opt_bin_val(x):
        """
        Freedman-Diaconis rule for optimal bin width.
        h = 2 * IQR(x) / n^(1/3)

        Matches R:  h <- 2 * IQR(x) / length(x)^(1/3)
        """
        n = len(x)
        if n < 2:
            return 1.0
        h = 2 * iqr(x) / max(n ** (1 / 3), 1)
        return float(h) if h > 0 else 1.0

    # ---------------------------------------------------------------------------
    # Hexagonal binning
    # ---------------------------------------------------------------------------

    def compute_hex_bins(layout_2d, num_bins=None):
        """
        Assign each 2D point to a hexagonal bin.

        Matches R: hexbin(x, y, num_bins, IDs=TRUE) + hcell2xy()

        Parameters
        ----------
        layout_2d : np.ndarray, shape (n, 2)
            2D NLDR embedding of the data.
        num_bins : int, optional
            Hexbin gridsize along the longest axis.
            Auto-computed via Freedman-Diaconis rule if None.

        Returns
        -------
        bin_centroids_2d : np.ndarray, shape (m, 2)
            2D centre coordinate of each occupied hexagon.
        bin_ids : np.ndarray, shape (n,)
            0-based bin index for each data point (nearest centroid).
        bin_counts : np.ndarray, shape (m,)
            Number of data points in each occupied bin.
        """
        x = layout_2d[:, 0]
        y = layout_2d[:, 1]
        n = len(x)

        if num_bins is None:
            bw_x = find_opt_bin_val(x)
            bw_y = find_opt_bin_val(y)
            range_x = float(np.ptp(x)) if np.ptp(x) > 0 else 1.0
            range_y = float(np.ptp(y)) if np.ptp(y) > 0 else 1.0
            bins_x = max(int(np.ceil(range_x / bw_x)), 3)
            bins_y = max(int(np.ceil(range_y / bw_y)), 3)
            num_bins = max(bins_x, bins_y)
            # Cap so bins aren't so small that most have 0-1 points
            num_bins = min(num_bins, max(3, int(np.sqrt(n) // 2)))

        num_bins = max(int(num_bins), 3)

        fig, ax = plt.subplots()
        hb = ax.hexbin(x, y, gridsize=num_bins)
        plt.close(fig)

        all_centroids = np.asarray(hb.get_offsets())  # (total_bins, 2)
        counts = np.asarray(hb.get_array())  # (total_bins,)

        # Keep only bins that contain at least one point
        occupied = counts > 0
        bin_centroids_2d = all_centroids[occupied]
        bin_counts = counts[occupied]

        if len(bin_centroids_2d) == 0:
            raise ValueError(
                "No occupied hexagonal bins found. "
                "Try passing a smaller num_bins value."
            )

        # Assign each point to its nearest occupied bin centroid
        tree = cKDTree(bin_centroids_2d)
        _, bin_ids = tree.query(layout_2d)  # (n,)

        return bin_centroids_2d, bin_ids, bin_counts

    # ---------------------------------------------------------------------------
    # High-dimensional centroids
    # ---------------------------------------------------------------------------

    def compute_bin_centroids_highd(high_d_data, bin_ids):
        """
        Compute the high-dimensional centroid (mean) of each occupied bin.

        Matches R:  df %>% group_by(hb_id) %>% summarise_all(mean)

        Parameters
        ----------
        high_d_data : np.ndarray, shape (n, p)
            High-dimensional data (e.g. normalised design variables).
        bin_ids : np.ndarray, shape (n,)
            0-based bin index for each point (from compute_hex_bins).

        Returns
        -------
        bin_centroids_highd : np.ndarray, shape (m, p)
            High-D mean of all points in each bin, in ascending bin-id order.
        """
        df = pd.DataFrame(high_d_data.astype(float))
        df['_bin_id'] = bin_ids
        grouped = df.groupby('_bin_id').mean()
        # grouped index = unique bin ids; columns = original feature columns
        return grouped.values  # (m, p)

    # ---------------------------------------------------------------------------
    # Triangulation
    # ---------------------------------------------------------------------------

    def triangulate_bin_centroids(bin_centroids_2d):
        """
        Delaunay triangulation of 2D bin centroids, returning all unique edges.

        Matches R:  tri.mesh(x, y)  +  triangles(tr1) -> node pairs

        Parameters
        ----------
        bin_centroids_2d : np.ndarray, shape (m, 2)

        Returns
        -------
        line_from : list[int]
        line_to : list[int]
            0-based indices into bin_centroids_2d for each triangulation edge.
            These index directly into the first m rows of the combined
            Langevitour data matrix (model centroids come first).
        """
        if len(bin_centroids_2d) < 3:
            return [], []

        tri = Delaunay(bin_centroids_2d)

        edges = set()
        for simplex in tri.simplices:
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                a, b = sorted([simplex[i], simplex[j]])
                edges.add((a, b))

        edges = sorted(edges)
        line_from = [e[0] for e in edges]
        line_to = [e[1] for e in edges]
        return line_from, line_to

    # ---------------------------------------------------------------------------
    # Edge distance + filtering
    # ---------------------------------------------------------------------------

    def cal_dist(bin_centroids_2d, line_from, line_to):
        """
        2D Euclidean distance for each triangulation edge.

        Matches R:  cal_dist(tr_from_to_df_coord)
        """
        return np.array([
            np.linalg.norm(bin_centroids_2d[f] - bin_centroids_2d[t])
            for f, t in zip(line_from, line_to)
        ])

    def filter_long_edges(bin_centroids_2d, line_from, line_to, cutoff=None):
        """
        Remove triangulation edges longer than `cutoff`.

        If cutoff is None, uses the automatic gap criterion from the paper:
        threshold = distance value just before the largest jump in the
        sorted distance distribution.

        Matches R:  get_langevitour_with_dist_criteria()
        """
        if not line_from:
            return line_from, line_to

        dists = cal_dist(bin_centroids_2d, line_from, line_to)

        if cutoff is None:
            sorted_dists = np.sort(dists)
            gaps = np.diff(sorted_dists)
            if len(gaps) > 0:
                # Threshold at the distance just before the largest gap
                cutoff = sorted_dists[np.argmax(gaps) + 1]
            else:
                return line_from, line_to

        keep = dists < cutoff
        line_from = [line_from[i] for i in range(len(line_from)) if keep[i]]
        line_to = [line_to[i] for i in range(len(line_to)) if keep[i]]
        return line_from, line_to

    # ---------------------------------------------------------------------------
    # Full pipeline
    # ---------------------------------------------------------------------------

    def build_hex_model(high_d_data, layout_2d, num_bins=None,
                        filter_edges=True, edge_cutoff=None):
        """
        Full pipeline: hexbin -> high-D centroids -> triangulation -> edges.

        Parameters
        ----------
        high_d_data : np.ndarray, shape (n, p)
            Original high-dimensional data points.
        layout_2d : np.ndarray, shape (n, 2)
            2D NLDR embedding used for binning (e.g. PCA, t-SNE, UMAP output).
        num_bins : int, optional
            Hexbin gridsize. Auto-computed if None.
        filter_edges : bool
            Remove long edges using the automatic gap criterion.
        edge_cutoff : float, optional
            Manual distance cutoff for edge filtering (overrides automatic).

        Returns
        -------
        bin_centroids_2d : np.ndarray, shape (m, 2)
        bin_centroids_highd : np.ndarray, shape (m, p)
        line_from : list[int]   (0-based into the m centroids)
        line_to : list[int]
        """
        high_d_data = np.asarray(high_d_data, dtype=float)
        layout_2d = np.asarray(layout_2d, dtype=float)

        bin_centroids_2d, bin_ids, bin_counts = compute_hex_bins(
            layout_2d, num_bins=num_bins
        )
        m = len(bin_centroids_2d)
        print(f"  Hex bins: {m} occupied  (gridsize={num_bins or 'auto'})")

        bin_centroids_highd = compute_bin_centroids_highd(high_d_data, bin_ids)

        line_from, line_to = triangulate_bin_centroids(bin_centroids_2d)
        print(f"  Triangulation edges (before filtering): {len(line_from)}")

        if filter_edges and len(line_from) > 0:
            line_from, line_to = filter_long_edges(
                bin_centroids_2d, line_from, line_to, cutoff=edge_cutoff
            )
            print(f"  Triangulation edges (after filtering):  {len(line_from)}")

        return bin_centroids_2d, bin_centroids_highd, line_from, line_to

    high_d_data = np.asarray(high_d_data, dtype=float)
    layout_2d = np.asarray(layout_2d, dtype=float)
    n, p = high_d_data.shape

    print("\nBuilding hexagonal bin model...")
    bin_centroids_2d, bin_centroids_highd, line_from, line_to = build_hex_model(
        high_d_data, layout_2d,
        num_bins=num_bins,
        filter_edges=filter_edges,
        edge_cutoff=edge_cutoff,
    )
    m = len(bin_centroids_highd)

    # Combined matrix: model centroids first (rows 0..m-1), data after (rows m..m+n-1)
    # Edge indices reference rows 0..m-1, which is correct since centroids come first.
    combined = np.vstack([bin_centroids_highd, high_d_data])

    if groups is None:
        data_groups = ["data"] * n
    else:
        data_groups = list(groups)
    all_groups = ["model"] * m + data_groups

    if column_names is None:
        column_names = [f"x{i+1}" for i in range(p)]

    print(f"\nCreating Langevitour with hex model...")
    print(f"  Points: {len(combined)} (model={m}, data={n})")
    print(f"  Edges:  {len(line_from)}")
    print(f"  Groups: {sorted(set(all_groups))}")

    tour = Langevitour(
        data=combined,
        group=all_groups,
        line_from=line_from,
        line_to=line_to,
        column_names=column_names,
        point_size=2.5,
    )

    tour.write_html(output_file)
    print(f"\n  Saved to: {output_file}")
    return tour

def plot_live(x, f, x_, xxx_, objectives):

    # give a binary mask of feasiblity, so if feas==1, then it is feasible.
    feas = np.array([int(fi[0] < 1e-3) for fi in f])

    # # Stack real and sample points together
    # x_xxx_ = np.vstack([x_, xxx_])
    #
    # # Reduce to 2D latent space
    # start = time.time()
    # x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)
    #
    # # Split back to x and xxx projections
    # x_len = x.shape[0]
    # if x_len < 30:
    #     raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")
    # x__1 = x_xxx_latent[:x_len, :]
    # x__ = TSNE(n_components=2).fit_transform(x__1)
    # xxx__ = x_xxx_latent[x_len:, :]
    # x_len = x.shape[0]
    # x__ = x_xxx_latent[:x_len, :]
    # end = time.time()
    # print(
    #     f'\nFAST latent embedding took: {end - start}s\nFor a more structured plot please use TSNE on x__1=PCA(x_xxx_) with Mahalanobis distance')
    #
    # # Train KNN on the latent projection
    # k = 1
    # f_hat_latent = KNeighborsClassifier(n_neighbors=k)
    # f_hat_latent.fit(x__, f.ravel())
    # f_hat_pred_latent = f_hat_latent.predict(xxx__)
    # f_hat_pred_latent = f_hat_pred_latent.reshape(-1, 1)

    x__,xxx__ = latent_pca(x_, xxx_)
    f_hat_pred_latent = latent_knn(x__,xxx__,f)

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

    # # Stack real and sample points together
    # x_xxx_ = np.vstack([x_, xxx_])
    #
    # # Reduce to 2D latent space
    # start = time.time()
    # x_xxx_latent = PCA(n_components=2).fit_transform(x_xxx_)
    #
    # # Split back to x and xxx projections
    # x_len = x.shape[0]
    # if x_len < 30:
    #     raise ValueError(f"TSNE needs a minimum of 30 points, or reduce the perplexity(k-neighbors) parameter")
    # x__1 = x_xxx_latent[:x_len, :]
    # x__ = TSNE(n_components=2).fit_transform(x__1)
    # # from MulticoreTSNE import MulticoreTSNE as TSNE
    # # tsne = TSNE(n_jobs=4) # n_jobs is number of cores (4=1.2x speedup)
    # # x__ = tsne(n_components=2).fit_transform(x__1)
    # end = time.time()
    # print(f'\nFAST latent embedding took: {end - start}s\nFor a more structured plot please use TSNE on x__1=PCA(x_xxx_) with Mahalanobis distance')

    x__ = latent_pca_tsne(x_, xxx_)

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

