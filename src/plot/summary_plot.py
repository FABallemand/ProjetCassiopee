import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from .metric_plot import plot_loss, plot_accuracy
from .tsne_plot import plot_tsne_2d, plot_tsne_3d

def plot_summary(dataset, input_size, classes, modalities,
                 transformation, crop_transformation,
                 nb_train_samples, nb_validation_samples, nb_test_samples,
                 batch_size, shuffle, drop_last,
                 device, model, debug,
                 loss_function, optimizer_type,
                 epochs, learning_rates,
                 early_stopping, patience, min_delta,
                 start_timestamp, stop_timestamp, run_epochs,
                 train_accuracies, train_losses,
                 validation_accuracies, validation_losses,
                 test_accuracy, test_confusion_matrix,
                 tsne_results_2d=None, tsne_results_3d=None, labels=None,
                 path="summary_plot.png"):
    # Create figure
    tsne_flag = tsne_results_2d is not None or tsne_results_3d is not None
    nb_rows = 4 if tsne_flag else 3
    fig, axs = plt.subplots(nb_rows, 2, figsize=(20, 20))

    # Plot relevant data about the dataset and data loaders
    data = [
        ["Dataset", dataset],
        ["Input size", input_size],
        ["Classes", classes if classes is not None else "all"],
        ["Modalities", modalities],
        ["Transformation", transformation],
        ["Crop transformation", crop_transformation],
        ["Train samples", nb_train_samples],
        ["Validation samples", nb_validation_samples],
        ["Test samples", nb_test_samples],
        ["Batch size", batch_size],
        ["Shuffle", shuffle],
        ["Drop last", drop_last]
    ]
    axs[0, 0].table(cellText=data, loc="center")
    axs[0, 0].axis("off")

    # Plot relevant data about the neural network and the training
    data = [
        ["Device", device],
        ["Model", model],
        ["Debug", debug],
        ["Loss function", loss_function],
        ["Optimizer", optimizer_type],
        ["Epochs", epochs],
        ["Learning rate(s)", learning_rates],
        ["Early stopping", early_stopping],
        ["Patience", patience if early_stopping else ""],
        ["Min delta", min_delta if early_stopping else ""],
        ["Start training", start_timestamp.strftime("%Y/%m/%d %H:%M:%S")],
        ["Stop training", stop_timestamp.strftime("%Y/%m/%d %H:%M:%S")],
        ["Run epochs", run_epochs],
        ["Learning rate(s)", learning_rates]
    ]
    axs[0, 1].table(cellText=data, loc="center")
    axs[0, 1].axis("off")

    # Compute total number of run epochs
    nb_epochs = sum(run_epochs)
    t = np.arange(nb_epochs)

    # Plot loss
    plot_loss(axs[1, 0], t, train_losses, validation_losses, run_epochs)

    # Plot accuracy
    if train_accuracies is not None and validation_accuracies is not None:
        plot_loss(axs[1, 1], t, train_accuracies, validation_accuracies, run_epochs)

    # Plot confusion matrix
    if test_confusion_matrix is not None:
        sns.heatmap(test_confusion_matrix, annot=True, cmap="flare",  fmt="d", cbar=True, ax=axs[2, 0])
        axs[2, 0].set_title(f"Test accuracy: {test_accuracy}")

    # Plot 2D TSNE
    if tsne_results_2d is not None:
        plot_tsne_2d(axs[3, 0], tsne_results_2d, labels)

    # Plot 3D TSNE
    if tsne_results_3d is not None:
        axs[3, 1].remove()
        axs[3, 1] = fig.add_subplot(4, 2, 8, projection="3d")
        plot_tsne_2d(axs[3, 1], tsne_results_2d, labels)

    plt.savefig(path)