import matplotlib.pyplot as plt


def plot_tsne_2d(ax, tsne_results_2d, labels):
    """
    Plot results of 2D TSNE.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to plot on
    tsne_results_2d : numpy.ndarray
        Results of 2D TSEN
    labels : numpy.ndarray
        Labels
    """
    ax.scatter(tsne_results_2d[:,0], tsne_results_2d[:,1], c=labels, s=50, alpha=0.8)


def plot_tsne_3d(ax, tsne_results_3d, labels):
    """
    Plot results of 3D TSNE.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to plot on (WARNING: must have projection set to 3D)
    tsne_results_2d : numpy.ndarray
        Results of 3D TSEN
    labels : numpy.ndarray
        Labels
    """
    ax.scatter(tsne_results_3d[:,0], tsne_results_3d[:,1], tsne_results_3d[:,2], c=labels, s=50, alpha=0.8)