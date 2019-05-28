from abc import abstractmethod
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from ..losses import Losses
from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Encoder:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.BATCH_SIZE = self.input_shape[0]

    @abstractmethod
    def _encoder(self):
        pass

    @abstractmethod
    def _decoder(self):
        pass

    @abstractmethod
    def get_compiled_model(self, *args):
        pass
