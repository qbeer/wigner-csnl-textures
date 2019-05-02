from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense
from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class AutoEncoder:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    @abstractmethod
    def _encoder(self):
        pass

    @abstractmethod
    def _decoder(self):
        pass

    def get_compiled_model(self, loss_fn=None, lr=1e-3, decay=5e-5):
        input_img = Input(shape=self.input_shape)
        encoder = self._encoder()
        decoder = self._decoder()

        encoded = encoder(input_img)
        latent = Dense(self.latent_dim)(encoded)
        decoded = decoder(latent)

        self.loss_fn = self._get_loss(loss_fn)

        model = Model(input_img, decoded)
        model.compile(optimizer=Adam(lr=lr, decay=decay), loss=self.loss_fn)
        return model, model

    def _get_loss(self, loss_fn):
        if loss_fn == None:
            return self._bernoulli
        elif loss_fn == "binary":
            return self._binary
        elif loss_fn == "normal":
            return self._normal
        elif loss_fn == "normalDiag":
            return self._normalDiag

    """
      For binarized input
    """

    def _binary(self, x_true, x_reco):
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true, logits=x_reco)

    def _bernoulli(self, x_true, x_reco):
        return -tf.reduce_mean(tfd.Bernoulli(x_reco)._log_prob(x_true))

    """
      For non binarized input.
    """

    def _normal(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.Normal(x_reco, scale=tf.Variable(0.001))._log_prob(x_true))

    def _normalDiag(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.MultivariateNormalDiag(x_reco, scale_identity_multiplier=tf.Variable(0.001))._log_prob(x_true))
