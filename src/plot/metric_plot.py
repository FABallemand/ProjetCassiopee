import matplotlib.pyplot as plt

def plot_loss(ax, epochs, train_loss, validation_loss=[], run_epochs=[]):
    # Plot loss over time
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, validation_loss, label="Validation loss")
    offset = 0
    for e in run_epochs:
        ax.axvline(x=e + offset, color="r", ls="--")
        offset += e
    ax.legend()


def loss_plot(epochs, train_loss, validation_loss, run_epochs, path):
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    plot_loss(ax, epochs, train_loss, validation_loss, run_epochs)
    plt.savefig(path)


def plot_accuracy(ax, epochs, train_accuracy, validation_accuracy=[], run_epochs=[]):
    # Plot accuracy over time
    ax.plot(epochs, train_accuracy, label="Train accuracy")
    ax.plot(epochs, validation_accuracy, label="Validation accuracy")
    offset = 0
    for e in run_epochs:
        ax.axvline(x=e + offset, color="r", ls="--")
        offset += e
    ax.legend()


def accuracy_plot(epochs, train_accuracy, validation_accuracy, run_epochs, path):
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    plot_accuracy(ax, epochs, train_accuracy, validation_accuracy, run_epochs)
    plt.savefig(path)


def loss_accuracy_plot(epochs, train_loss, train_accuracy, validation_loss, validation_accuracy, run_epochs, path):
    fig, axs = plt.subplots(1, 2)
    plot_loss(axs[0], epochs, train_loss, validation_loss, run_epochs)
    plot_accuracy(axs[1], epochs, train_accuracy, validation_accuracy, run_epochs)
    plt.savefig(path)


def confusion_matrix_plot():
    pass