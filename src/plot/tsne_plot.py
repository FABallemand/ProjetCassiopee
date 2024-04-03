import matplotlib.pyplot as plt


def plot_tsne_2d(ax, tsne_results_2d, labels):
    """
    Plot results of 2D TSNE on subplot.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to plot on
    tsne_results_2d : numpy.ndarray
        Results of 2D TSNE
    labels : numpy.ndarray
        Labels
    """
    ax.scatter(tsne_results_2d[:,0], tsne_results_2d[:,1], c=labels, s=50, alpha=0.8)


def tsne_2d_plot(tsne_results_2d, labels, path=None):
    """
    Create plot with results of 2D TSNE.

    Parameters
    ----------
    tsne_results_2d : numpy.ndarray
        Results of 2D TSNE
    labels : numpy.ndarray
        Labels
    path : str
        Path to save the plot, by default None
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plot_tsne_2d(ax, tsne_results_2d, labels)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_tsne_3d(ax, tsne_results_3d, labels):
    """
    Plot results of 3D TSNE on subplot.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to plot on (WARNING: must have projection set to 3D)
    tsne_results_2d : numpy.ndarray
        Results of 3D TSNE
    labels : numpy.ndarray
        Labels
    """
    ax.scatter(tsne_results_3d[:,0], tsne_results_3d[:,1], tsne_results_3d[:,2], c=labels, s=50, alpha=0.8)


def tsne_3d_plot(tsne_results_3d, labels, path=None):
    """
    Create plot with results of 3D TSNE.

    Parameters
    ----------
    tsne_results_3d : numpy.ndarray
        Results of 3D TSNE
    labels : numpy.ndarray
        Labels
    path : str
        Path to save the plot, by default None
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.remove()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_tsne_3d(ax, tsne_results_3d, labels)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def tsne_plot(tsne_results_2d, tsne_results_3d, labels, path=None):
    """
    Create plot with results of 2D and 3D TSNE.

    Parameters
    ----------
    tsne_results_2d : numpy.ndarray
        Results of 2D TSNE
    tsne_results_3d : numpy.ndarray
        Results of 3D TSNE
    labels : numpy.ndarray
        Labels
    path : str
        Path to save the plot, by default None
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    plot_tsne_2d(axs[0], tsne_results_2d, labels)
    axs[1].remove()
    axs[1] = fig.add_subplot(1, 2, 2, projection='3d')
    plot_tsne_3d(axs[1], tsne_results_3d, labels)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)