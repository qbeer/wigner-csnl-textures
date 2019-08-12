from .small_dense_lvae import SmallDenseLadderVAE
from keras.layers import Input, Dense, ReLU
from keras.models import Model


class SmallDenseLinearLadderVAE(SmallDenseLadderVAE):
    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(256)(input_tensor)
        x = Dense(self.latent_dim1)(x)
        encoder = Model(input_tensor, x, name="linear_encoder")
        return encoder
