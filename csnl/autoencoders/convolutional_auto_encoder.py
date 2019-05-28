from ..encoder import ConvolutionalEncoder
from .auto_encoder import AutoEncoder
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, PReLU, Dense, Flatten, Reshape
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense
from ..encoder import Encoder
from ..losses import Losses
from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class ConvolutionalAutoEncoder(ConvolutionalEncoder, AutoEncoder):
    pass
