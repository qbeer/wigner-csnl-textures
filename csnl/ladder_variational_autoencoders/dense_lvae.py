from .ladder_vae import LadderVAE
from keras.layers import Input, Dense, ReLU
from keras.models import Model


class DenseLadderVAE(LadderVAE):
    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(1024)(input_tensor)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        x = Dense(self.latent_dim1)(x)
        encoder = Model(input_tensor, x)
        return encoder

    def encoder2(self):
        input_tensor = Input(shape=(self.latent_dim1,))
        x = Dense(512)(input_tensor)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        encoder = Model(input_tensor, x)
        return encoder

    def decoder2(self):
        latent2 = Input(shape=(self.latent_dim2,))
        x = Dense(256)(latent2)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        x = Dense(self._mean_variance_input_shape)(x)
        reco = ReLU()(x)
        decoder = Model(latent2, reco)
        return decoder

    def decoder1(self):
        latent1 = Input(shape=(self.latent_dim1,))
        x = Dense(512)(latent1)
        x = ReLU()(x)
        x = Dense(1024)(x)
        x = ReLU()(x)
        x = Dense(2048)(x)
        x = ReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent1, reco)
        return decoder
