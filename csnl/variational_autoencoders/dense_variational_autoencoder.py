from .variational_autoencoder import VariationalAutoEncoder
from keras.layers import Input, Dense, PReLU
from keras.models import Model


class DenseVAE(VariationalAutoEncoder):

    def _encoder(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(512)(input_tensor)
        x = PReLU()(x)
        x = Dense(256)(x)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
        encoder = Model(input_tensor, x)
        return encoder

    def _decoder(self):
        latent = Input(shape=(self.latent_dim,))
        x = Dense(256)(latent)
        x = PReLU()(x)
        x = Dense(512)(x)
        x = PReLU()(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent, reco)
        return decoder
