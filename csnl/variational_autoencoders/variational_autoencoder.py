from keras.optimizers import Adam, RMSprop
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import tensorflow as tf
from abc import abstractmethod
import keras.backend as K
import tensorflow_probability as tfp
from ..encoder import Encoder
tfd = tfp.distributions


class VariationalAutoEncoder(Encoder):
    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.BATCH_SIZE, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_compiled_model(self, *args):
        loss_fn, lr, decay, self.observation_noise, beta = args
        input_img = Input(batch_shape=self.input_shape)

        encoder = self._encoder()
        encoded = encoder(input_img)

        # Reparametrization
        self.z_mean = Dense(self.latent_dim)(encoded)
        self.z_log_sigma = Dense(self.latent_dim)(encoded)
        z = Lambda(self._sampling, name="latent")([self.z_mean, self.z_log_sigma])

        decoder = self._decoder()
        reco = decoder(z)

        self.beta = beta
        self.loss_fn = self._get_loss(loss_fn)

        # Generator model
        decoder_input = Input(shape=(self.latent_dim,))
        _reco = decoder(decoder_input)
        generator = Model(decoder_input, _reco)

        # Model for later inference
        self.latent_model = Model(input_img, outputs=[reco, z])

        model = Model(input_img, reco)
        model.compile(optimizer=RMSprop(lr=lr, decay=decay),
                      loss=self.loss_fn, metrics=[self.KL_divergence])

        return model, generator