from .ladder_vae import LadderVAE
from keras.layers import Input, Dense, ReLU, BatchNormalization
from keras.models import Model


class SmallDenseLadderVAE_BN(LadderVAE):
    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(512)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(self.latent_dim1)(x)
        encoder = Model(input_tensor, x)
        return encoder

    def encoder2(self):
        input_tensor = Input(shape=(self.latent_dim1,))
        x = Dense(256)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        encoder = Model(input_tensor, x)
        return encoder

    def decoder2(self):
        latent2 = Input(shape=(self.latent_dim2,))
        x = Dense(256)(latent2)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(self._mean_variance_input_shape)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        decoder = Model(latent2, x)
        return decoder

    def decoder1(self):
        latent1 = Input(shape=(self.latent_dim1,))
        x = Dense(256)(latent1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(self.input_shape[1])(x)
        decoder = Model(latent1, x)
        return decoder
