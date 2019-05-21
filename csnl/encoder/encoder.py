from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense
from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K
tfd = tfp.distributions


class Encoder:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.BATCH_SIZE = self.input_shape[0]

    @abstractmethod
    def _encoder(self):
        pass

    @abstractmethod
    def _decoder(self):
        pass

    @abstractmethod
    def get_compiled_model(self, *args):
        pass

    def _get_loss(self, loss_fn):
        losses = {"binary": self._binary,
                  "normal": self._normal,
                  "normalDiag": self._normalDiag,
                  None: self._bernoulli}
        return losses[loss_fn]

    """
      Making it custom metric to be able to feed it to Keras API - actually no need for y_true, y_pred
    """

    def KL_divergence(self, y_true, y_pred):
        if self.beta > 0:
            return - self.beta * 0.5 * K.mean(
                1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma), axis=-1)
        return 0
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
