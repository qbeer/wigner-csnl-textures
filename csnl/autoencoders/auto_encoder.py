from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense
from ..encoder import Encoder
from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class AutoEncoder(Encoder):
    def get_compiled_model(self, *args):
        loss_fn, lr, decay, self.observation_noise, self.beta = args
        self.beta = 0  # no KL loss (!)
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

        self.latent_model = Model(input_img, outputs=[decoded, latent])

        self.loss_fn = self._get_loss(loss_fn)

        model = Model(input_img, decoded)
        model.compile(optimizer=Adam(lr=lr, decay=decay), loss=self.loss_fn)
        return model, generative_model
