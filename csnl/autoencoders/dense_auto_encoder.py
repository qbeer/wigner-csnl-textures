from ..encoder import DenseEncoder
from ..losses import Losses
from .auto_encoder import AutoEncoder
from keras.layers import Input, Dense, PReLU
from keras.models import Model
from keras.optimizers import Adam


class DenseAutoEncoder(DenseEncoder, AutoEncoder):
    pass
