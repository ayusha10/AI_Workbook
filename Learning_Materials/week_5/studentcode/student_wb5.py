from approvedimports import *

def cluster_and_visualise(datafile_name: str, K: int, feature_names: list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use
        
        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot
        
        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
    # Suppress the memory leak warning on Windows with MKL
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # get the data from file into a numpy array
    data = np.genfromtxt(datafile_name, delimiter=',', filling_values=np.nan)

    # Validate data shape
    n_features = len(feature_names)
    if data.ndim != 2 or data.shape[1] != n_features:
        raise ValueError(f"Data has {data.shape[1] if data.ndim == 2 else 'invalid'} columns, but {n_features} feature names were provided.")

    # create a K-Means cluster model with the specified number of clusters
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(data)

    # Define colors and markers for clusters (using highly distinct colors and markers)
    colors = ['red', 'blue', 'lime', 'purple', 'orange', 'cyan', 'magenta'][:K]
    markers = ['o', '^', 's', 'D', 'v', 'p', '*'][:K]

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    if n_features == 1:  # Handle case where there's only 1 feature
        axes = np.array([[axes]])

    # Plot scatter plots for each pair of features and histograms on the diagonal
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:  # Diagonal: Plot histogram (no labels)
                for k in range(K):
                    cluster_data = data[cluster_ids == k, i]
                    cluster_data = cluster_data[~np.isnan(cluster_data)]
                    if len(cluster_data) > 0:  # Only plot if there are valid data points
                        ax.hist(cluster_data, bins=15, alpha=0.7, color=colors[k])
            else:  # Off-diagonal: Plot scatter
                for k in range(K):
                    cluster_data = data[cluster_ids == k]
                    mask = ~np.isnan(cluster_data[:, j]) & ~np.isnan(cluster_data[:, i])
                    if np.sum(mask) > 0:  # Only plot if there are valid data points
                        ax.scatter(cluster_data[mask, j], cluster_data[mask, i], 
                                  c=colors[k], marker=markers[k], s=40, alpha=0.4, 
                                  label=f'Cluster {k}', edgecolors='k', linewidth=0.5)
                if j == 0:  # Leftmost column only
                    ax.set_ylabel(feature_names[i])
                if i == n_features - 1:  # Bottom row only
                    ax.set_xlabel(feature_names[j])

    # Collect handles and labels from scatter plots
    handles, labels = [], []
    seen_labels = set()
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                h, l = axes[i, j].get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in seen_labels:
                        handles.append(hh)
                        labels.append(ll)
                        seen_labels.add(ll)

    # Add a single legend for the entire figure
    if handles:  # Only add legend if there are handles
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                  title="Clusters", fontsize=12, frameon=True)

    # Adjust layout and set the title
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.suptitle(f'Visualisation of {K} clusters by a5-poudel', y=1.0, fontsize=16)

    # Save to file
    fig.savefig('myVisualisation.jpg', dpi=300, bbox_inches='tight')

    # Return the figure and axes
    return fig, axes
