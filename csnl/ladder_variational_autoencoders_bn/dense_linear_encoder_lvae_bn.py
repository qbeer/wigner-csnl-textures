from .dense_lvae_bn import DenseLadderVAE_BN
from keras.layers import Input, Dense, ReLU, BatchNormalization
from keras.models import Model


class DenseLinearLadderVAE_BN(DenseLadderVAE_BN):
    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(256)(input_tensor)
        x = Dense(self.latent_dim1)(x)
        x = BatchNormalization()(x)
        encoder = Model(input_tensor, x, name="dense_linear_encoder")
        return encoder
