from .variational_autoencoder import VariationalAutoEncoder
from keras.layers import Input, Dense, ReLU
from keras.models import Model


class SmallDenseVAE(VariationalAutoEncoder):

    def _encoder(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(420)(input_tensor)
        x = ReLU()(x)
        x = Dense(210)(x)
        x = ReLU()(x)
        x = Dense(105)(x)
        x = ReLU()(x)
        encoder = Model(input_tensor, x, name="small_dense_encoder")
        return encoder

    def _decoder(self):
        latent = Input(shape=(self.latent_dim,))
        x = Dense(200)(latent)
        x = ReLU()(x)
        x = Dense(400)(x)
        x = ReLU()(x)
        x = Dense(800)(x)
        x = ReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent, reco, name="small_dense_decoder")
        return decoder
