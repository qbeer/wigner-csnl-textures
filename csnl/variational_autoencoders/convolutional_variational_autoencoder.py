from ..encoder import ConvolutionalEncoder
from keras.layers import Input, Conv2D, PReLU, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
from keras.optimizers import Adam, RMSprop
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import tensorflow as tf
from abc import abstractmethod
import keras.backend as K
import tensorflow_probability as tfp
from .variational_autoencoder import VariationalAutoEncoder
tfd = tfp.distributions


class ConvolutionalVAE(ConvolutionalEncoder, VariationalAutoEncoder):
    pass
