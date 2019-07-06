from keras.callbacks import Callback
import keras.backend as K


class IncrementalBeta(Callback):
    def __init__(self, beta, n_epochs=20):
        self.beta_max = beta
        self.n_epochs = n_epochs
        self.beta = 0

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.model.beta, self.beta)

    def on_epoch_end(self, epoch, logs=None):
        updated_beta = self.beta + self.beta_max / (3 * self.n_epochs // 4)
        if updated_beta < self.beta_max:
            self.beta = updated_beta
            print("Beta updated : ", self.beta)
        else:
            self.beta = self.beta_max
