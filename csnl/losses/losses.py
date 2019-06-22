import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K
tfd = tfp.distributions


class Losses:
    def __init__(self, loss_fn, observation_noise, beta=0,
                 z_mean=None, z_sigma=None, z2_mean=None, z2_log_sigma=None,
                 z_mean_TD=None, z_log_sigma_TD=None):
        self.loss = self._get_loss(loss_fn)
        self.beta = beta
        self.observation_noise = observation_noise
        self.z2_mean = z2_mean
        self.z2_log_sigma = z2_log_sigma
        if self.beta > 0:
            assert z2_mean != None, "z_mean should be defined!"
            assert z2_log_sigma != None, "z_log_sigma should be defined"
        self.z_mean = z_mean
        self.z_sigma = z_sigma
        self.z_mean_TD = z_mean_TD
        self.z_log_sigma_TD = z_log_sigma_TD
        if self.z_mean != None and self.z_sigma != None:
            assert z_mean_TD != None, "z_mean_TD should be defined!"
            assert z_log_sigma_TD != None, "z_log_sigma_TD should be defined!"

    def _get_loss(self, loss_fn):
        losses = {"binary": self._binary,
                  "normal": self._normal,
                  "normalDiag": self._normalDiag,
                  "bernoulli": self._bernoulli}
        return losses[loss_fn]

    """
      Making it custom metric to be able to feed it to Keras API - actually no need for y_true, y_pred
    """

    def KL_divergence(self, y_true, y_pred):
        if self.beta > 0:
            p2 = tfd.Normal(self.z2_mean, tf.exp(self.z2_log_sigma) + 1e-12)
            q2 = tfd.Normal(0, 1)
            if self.z_mean != None and self.z_sigma != None:
                p = tfd.Normal(self.z_mean, self.z_sigma + 1e-12)
                q = tfd.Normal(self.z_mean_TD, tf.exp(
                    self.z_log_sigma_TD) + 1e-12)
                return self.beta * (tf.reduce_mean(tfd.kl_divergence(p, q)) + tf.reduce_mean(tfd.kl_divergence(p2, q2)))
            else:
                return self.beta * tf.reduce_mean(tfd.kl_divergence(p2, q2))

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
