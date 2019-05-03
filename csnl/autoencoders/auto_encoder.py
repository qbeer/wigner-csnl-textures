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

    def get_compiled_model(self, *args):
        loss_fn, lr, decay, self.observation_noise, _ = args
        input_img = Input(shape=self.input_shape)
        encoder = self._encoder()
        decoder = self._decoder()

        encoded = encoder(input_img)
        latent = Dense(self.latent_dim)(encoded)
        decoded = decoder(latent)

        # Generative model
        latent_input = Input(shape=(self.latent_dim, ))
        _decoded = decoder(latent_input)
        generative_model = Model(latent_input, _decoded)

        self.loss_fn = self._get_loss(loss_fn)

        model = Model(input_img, decoded)
        model.compile(optimizer=Adam(lr=lr, decay=decay), loss=self.loss_fn)
        return model, generative_model

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
            tfd.Normal(x_reco, scale=self.observation_noise)._log_prob(x_true))

    def _normalDiag(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.MultivariateNormalDiag(x_reco, scale_identity_multiplier=self.observation_noise)._log_prob(x_true))
