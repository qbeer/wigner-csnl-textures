from keras.optimizers import Adam, RMSprop
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import tensorflow as tf
from abc import abstractmethod
import keras.backend as K
import tensorflow_probability as tfp
from ..encoder import Encoder
from ..losses import Losses
tfd = tfp.distributions


class VariationalAutoEncoder(Encoder):
    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.BATCH_SIZE, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_compiled_model(self, *args):
        loss_fn, lr, decay, self.observation_noise, self.beta = args
        input_img = Input(batch_shape=self.input_shape)

        encoder = self._encoder()
        decoder = self._decoder()

        encoded = encoder(input_img)

        # Reparametrization
        self.z_mean = Dense(self.latent_dim, name="mean")(encoded)
        self.z_log_sigma = Dense(self.latent_dim, name="log_sigma")(encoded)
        z = Lambda(self._sampling, name="sampling_z")(
            [self.z_mean, self.z_log_sigma])

        reco = decoder(z)

        # Generator model
        decoder_input = Input(shape=(self.latent_dim,))
        _reco = decoder(decoder_input)
        generator = Model(decoder_input, _reco)

        # Model for later inference
        latent_model = Model(input_img, outputs=[reco, z])

        model = Model(input_img, reco)
        model.beta = K.variable(self.beta)

        losses = Losses(loss_fn, self.observation_noise, model.beta,
                        z2_mean=self.z_mean, z2_log_sigma=self.z_log_sigma)

        model.compile(optimizer=RMSprop(lr=lr, decay=decay),
                      loss=losses.loss, metrics=[losses.KL_divergence])

        return model, generator, latent_model
