import numpy as np
import torch


class EarlyStopping:
    """Monitor a metric and stop training when it stops improving.
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            trace_func (function): Trace print function. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):       
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping: valid_loss({self.best_score:.6f} to {score:.6f}) increased, counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ModelCheckpoint:
    """Automatically save model checkpoints during training.
    """

    def __init__(self, path='checkpoint.pt', monitor='val_loss', verbose=False, trace_func=print):
        """
        Args:
            path (float): Path to save the model file. Default: 'checkpoint.pt'
            monitor (str): The metric name to monitor. Default: 'val_loss'
            verbose (bool): Displays messages when the callback takes an action. Default: False
            trace_func (function): Trace print function. Default: print
        """
        self.path = path
        self.monitor = monitor
        self.verbose = verbose
        self.trace_func = trace_func

        if self.monitor == 'val_loss':
            self.best_score = np.inf
        elif self.monitor == 'val_acc':
            self.best_score = -np.inf

    def __call__(self, score, model):
        if (self.best_score == np.inf) or (self.best_score == -np.inf):
            self.save_checkpoint(score, model)
            self.best_score = score
        else:
            if self.monitor == 'val_loss':
                if score < self.best_score:
                    self.save_checkpoint(score, model)
                    self.best_score = score
            elif self.monitor == 'val_acc':
                if score > self.best_score:
                    self.save_checkpoint(score, model)
                    self.best_score = score

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.trace_func(
                f'ModelCheckpoint: {self.monitor}({self.best_score:.6f} to {score:.6f}) decreased, saving model...'
            )
        torch.save(model.state_dict(), self.path)
