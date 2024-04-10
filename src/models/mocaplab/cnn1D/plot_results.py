import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(train_accuracies, train_losses,
                 validation_accuracies, validation_losses,
                 run_epochs, architecture, start_timestamp, device,
                 loss_function, optimizer_type, epochs,
                 learning_rates, early_stopping, patience, min_delta,
                 test_accuracy, test_confusion_matrix,
                 stop_timestamp, model_path):
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    # Compute total number of run epochs
    nb_epochs = sum(run_epochs)
    t = np.arange(nb_epochs)

    # Plot accuracy over time
    axs[0, 0].plot(t, train_accuracies, label="Train accuracy")
    axs[0, 0].plot(t, validation_accuracies, label="Validation accuracy")
    offset = 0
    for e in run_epochs:
        axs[0, 0].axvline(x=e + offset, color="r", ls="--")
        offset += e
    axs[0, 0].legend()

    # Plot loss over time
    axs[0, 1].plot(t, train_losses, label="Train loss")
    axs[0, 1].plot(t, validation_losses, label="Validation loss")
    offset = 0
    for e in run_epochs:
        axs[0, 1].axvline(x=e + offset, color="r", ls="--")
        offset += e
    axs[0, 1].legend()

    # Plot confusion matrix
    sns.heatmap(test_confusion_matrix, annot=True, cmap="flare",  fmt="d", cbar=True, ax=axs[1, 0])

    # Plot relevant data about the neural network and the training
    data = [
        ["Architecture", architecture],
        ["Start training", start_timestamp.strftime("%Y/%m/%d %H:%M:%S")],
        ["Device", device],
        ["Loss function", loss_function],
        ["Optimizer", optimizer_type],
        ["Epochs", epochs],
        ["Learning rate(s)", learning_rates],
        ["Run epochs", run_epochs if early_stopping else ""],
        ["Patience", patience if early_stopping else ""],
        ["Min delta", min_delta if early_stopping else ""],
        ["Test accuracy", test_accuracy],
        ["Stop training", stop_timestamp.strftime("%Y/%m/%d %H:%M:%S")]
    ]

    axs[1, 1].table(cellText=data, loc="center")
    axs[1, 1].axis("off")

    # Save figure
    plt.savefig("self_supervised_learning/dev/ProjetCassiopee/train_results/fc_mocaplab/" + model_path + ".png")

    plt.show()