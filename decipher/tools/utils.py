import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Parameters
    ----------
    patience : int
        How long to wait after last time validation loss improved.

    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
