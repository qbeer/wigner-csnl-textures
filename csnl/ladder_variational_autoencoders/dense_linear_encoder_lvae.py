from .dense_lvae import DenseLadderVAE
from keras.layers import Input, Dense, ReLU
from keras.models import Model


class DenseLinearLadderVAE(DenseLadderVAE):
    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(256)(input_tensor)
        x = Dense(self.latent_dim1)(x)
        encoder = Model(input_tensor, x, name="dense_linear_encoder")
        return encoder
