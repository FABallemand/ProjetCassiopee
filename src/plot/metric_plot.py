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


def plot_accuracy(ax, epochs, train_acc, validation_acc=[], run_epochs=[]):
    # Plot accuracy over time
    ax.plot(epochs, train_acc, label="Train accuracy")
    ax.plot(epochs, validation_acc, label="Validation accuracy")
    offset = 0
    for e in run_epochs:
        ax.axvline(x=e + offset, color="r", ls="--")
        offset += e
    ax.legend()