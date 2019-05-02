from keras.optimizers import Adam
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import tensorflow as tf
from abc import abstractmethod
import keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions


class VariationalAutoEncoder:
    def __init__(self, input_shape, latent_dim, BATCH_SIZE):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.BATCH_SIZE = BATCH_SIZE

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
        print(args)
        loss_fn, lr, decay, beta = args
        input_img = Input(batch_shape=(self.BATCH_SIZE, *self.input_shape))

        encoder = self._encoder()
        encoded = encoder(input_img)

        # Reparametrization
        self.z_mean = Dense(self.latent_dim)(encoded)
        self.z_log_sigma = Dense(self.latent_dim)(encoded)
        z = Lambda(self._sampling)([self.z_mean, self.z_log_sigma])

        decoder = self._decoder()
        reco = decoder(z)

        self.beta = beta
        self.loss_fn = self._get_loss(loss_fn)

        # Generator model
        decoder_input = Input(shape=(self.latent_dim,))
        _reco = decoder(decoder_input)
        generator = Model(decoder_input, _reco)

        model = Model(input_img, reco)
        model.compile(optimizer=Adam(lr=lr, decay=decay), loss=self.loss_fn)

        return model, generator

    """
      For metrics (!)
    """

    def _kl_loss(self):
        return - self.beta * 0.5 * tf.reduce_mean(
            1 + self.z_log_sigma - tf.square(self.z_mean) - tf.exp(self.z_log_sigma), axis=-1)

    """
      For binarized input with KL term (!)
    """

    def _binary(self, x_true, x_reco):
        return -tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x_true, logits=x_reco) + self._kl_loss()

    def _bernoulli(self, x_true, x_reco):
        return -tf.reduce_mean(tfd.Bernoulli(x_reco)._log_prob(x_true)
                               ) + self._kl_loss()

    """
      For non binarized input with KL term(!)
    """

    def _normal(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.Normal(x_reco, scale=tf.Variable(0.001))._log_prob(x_true)
        ) + self._kl_loss()

    def _normalDiag(self, x_true, x_reco):
        return -tf.reduce_mean(
            tfd.MultivariateNormalDiag(
                x_reco, scale_identity_multiplier=tf.Variable(0.001))._log_prob(x_true)) + self._kl_loss()
