from .auto_encoder import AutoEncoder
from keras.layers import Input, Dense, PReLU
from keras.models import Model
from keras.optimizers import Adam


class DenseAutoEncoder(AutoEncoder):

    def _encoder(self):
        input_img = Input(shape=self.input_shape)
        x = Dense(512)(input_img)
        x = PReLU()(x)
        x = Dense(256)(x)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
        encoder = Model(input_img, x)
        return encoder

    def _decoder(self):
        latent_input = Input(shape=(self.latent_dim, ))
        x = Dense(256)(latent_input)
        x = PReLU()(x)
        x = Dense(512)(x)
        x = PReLU()(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        reco = Dense(self.input_shape[0])(x)
        decoder = Model(latent_input, reco)
        return decoder
