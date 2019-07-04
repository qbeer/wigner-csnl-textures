from .encoder import Encoder
from keras.layers import Input, Dense, ReLU
from keras.models import Model
from keras.optimizers import Adam


class DenseEncoder(Encoder):

    def _encoder(self):
        input_img = Input(shape=self.input_shape[1:])
        x = Dense(1024)(input_img)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        encoder = Model(input_img, x, name="dense_encoder")
        return encoder

    def _decoder(self):
        latent_input = Input(shape=(self.latent_dim, ))
        x = Dense(512)(latent_input)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        x = Dense(1024)(x)
        x = ReLU()(x)
        x = Dense(1024)(x)
        x = ReLU()(x)
        x = Dense(2048)(x)
        x = ReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent_input, reco, name="dense_decoder")
        return decoder
