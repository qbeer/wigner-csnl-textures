from ..encoder import DenseEncoder
from keras.layers import Input, Dense, PReLU
from keras.optimizers import Adam, RMSprop
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import tensorflow as tf
from abc import abstractmethod
import keras.backend as K
import tensorflow_probability as tfp
from ..encoder import Encoder
from ..losses import Losses
from .variational_autoencoder import VariationalAutoEncoder
tfd = tfp.distributions


class DenseVAE(DenseEncoder, VariationalAutoEncoder):
    pass
