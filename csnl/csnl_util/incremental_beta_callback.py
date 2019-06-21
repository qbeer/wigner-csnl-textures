from keras.callbacks import Callback


class IncrementalBeta(Callback):
    def __init__(self, beta, n_epochs=20):
        self.beta_max = beta
        self.n_epochs = n_epochs
        self.beta = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.beta < self.beta_max:
            self.beta += self.beta_max / self.n_epochs
        else:
            self.beta = 1
