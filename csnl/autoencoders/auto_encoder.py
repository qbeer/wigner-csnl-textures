from ..encoder import Encoder
from ..losses import Losses
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


class AutoEncoder(Encoder):
    def get_compiled_model(self, *args):
        loss_fn, lr, decay, observation_noise, _ = args
        input_img = Input(batch_shape=self.input_shape)
        encoder = self._encoder()
        decoder = self._decoder()

        encoded = encoder(input_img)
        latent = Dense(self.latent_dim)(encoded)
        decoded = decoder(latent)

        # Generative model
        latent_input = Input(shape=(self.latent_dim, ))
        _decoded = decoder(latent_input)
        generative_model = Model(latent_input, _decoded)

        latent_model = Model(input_img, outputs=[decoded, latent])

        losses = Losses(loss_fn, observation_noise)

        model = Model(input_img, decoded)
        model.compile(optimizer=Adam(lr=lr, decay=decay), loss=losses.loss)
        return model, generative_model, latent_model
