from keras.optimizers import Adam, RMSprop
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import tensorflow as tf
from abc import abstractmethod
import keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions


class VariationalAutoEncoder:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.BATCH_SIZE = input_shape[0]

    @abstractmethod
    def _encoder(self):
        pass

    @abstractmethod
    def _decoder(self):
        pass

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.BATCH_SIZE, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _get_loss(self, loss_fn):
        if loss_fn == None:
            return self._bernoulli
        elif loss_fn == "binary":
            return self._binary
        elif loss_fn == "normal":
            return self._normal
        elif loss_fn == "normalDiag":
            return self._normalDiag

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

        model = Model(input_img, outputs=[reco, z])
        model.compile(optimizer=RMSprop(lr=lr, decay=decay),
                      loss=self.loss_fn, metrics=[self.KL_divergence])

        return model, generator

    """
      Making it custom metric to be able to feed it to Keras API - actually no need for y_true, y_pred
    """

    def KL_divergence(self, y_true, y_pred):
        return - self.beta * 0.5 * K.mean(
            1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma), axis=-1)

    """
      For binarized input with KL term (!)
    """

    def _binary(self, x_true, x_reco):
        return -tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x_true, logits=x_reco) + self.KL_divergence(None, None)

    def _bernoulli(self, x_true, x_reco):
        return -tf.reduce_mean(tfd.Bernoulli(x_reco)._log_prob(x_true)
                               ) + self.KL_divergence(None, None)

    """
      For non binarized input with KL term(!)
    """

    def _normal(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.Normal(x_reco, scale=self.observation_noise)._log_prob(x_true)
        ) + self.KL_divergence(None, None)

    def _normalDiag(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.MultivariateNormalDiag(
                x_reco, scale_identity_multiplier=self.observation_noise)._log_prob(x_true)) + self.KL_divergence(None, None)
