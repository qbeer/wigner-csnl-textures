from keras.callbacks import Callback
import keras.backend as K


class IncrementalBeta(Callback):
    def __init__(self, beta, n_epochs=20):
        self.beta_max = beta
        self.n_epochs = n_epochs
        self.beta = 0

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.model.beta, self.beta_max if self.beta > self.beta_max else self.beta)

    def on_epoch_end(self, epoch, logs=None):
        if self.beta < self.beta_max:
            self.beta += self.beta_max / (self.n_epochs // 2)
            print("BETA updated : ", self.beta)
        else:
            self.beta = self.beta_max
