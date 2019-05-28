from abc import abstractmethod
from keras.layers import Input, Dense, PReLU, Lambda
from keras.models import Model
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class LadderVAE:
    def __init__(self, input_shape, latent_dim1, latent_dim2):
        self.input_shape = input_shape
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.BATCH_SIZE = self.input_shape[0]

    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(512)(input_tensor)
        x = PReLU()(x)
        x = Dense(256)(x)
        x = PReLU()(x)
        x = Dense(256)(x)
        x = PReLU()(x)
        x = Dense(self.latent_dim1 * 2)(x)
        encoder = Model(input_tensor, x)
        return encoder

    def encoder2(self):
        input_tensor = Input(shape=(self.latent_dim1 * 2,))
        x = Dense(256)(input_tensor)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
        x = Dense(self.latent_dim2 * 2)(x)
        encoder = Model(input_tensor, x)
        return encoder

    def decoder2(self):
        latent2 = Input(shape=(self.latent_dim2,))
        x = Dense(128)(latent2)
        x = PReLU()(x)
        x = Dense(256)(x)
        x = PReLU()(x)
        x = Dense(512)(x)
        x = PReLU()(x)
        reco = Dense(self.latent_dim1)(x)
        decoder = Model(latent2, reco)
        return decoder

    def decoder1(self):
        latent1 = Input(shape=(self.latent_dim1 * 2,))
        x = Dense(256)(latent1)
        x = PReLU()(x)
        x = Dense(512)(x)
        x = PReLU()(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent1, reco)
        return decoder

    def _importance_weight(self, args):
        input1, input2 = args
        res = tf.concat([input1, input2], axis=1)
        return res

    def _split(self, _input):
        return tf.split(axis=1, value=_input, num_or_size_splits=2)

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.BATCH_SIZE, self.latent_dim2 if self.mean ==
                   None else self.latent_dim1),
            mean=0 if self.mean == None else self.mean, stddev=1 if self.var == None else tf.sqrt(self.var))
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_compiled_model(self, *args):
        _, lr, decay, self.observation_noise, beta = args
        input_img = Input(batch_shape=self.input_shape)

        encoder1 = self.encoder1()
        encoder2 = self.encoder2()
        decoder1 = self.decoder1()
        decoder2 = self.decoder2()

        d1 = encoder1(input_img)

        d2 = encoder2(d1)

        # Reparametrization 1 & 2
        self.z2_mean, self.z2_log_sigma = Lambda(self._split)(d2)

        self.mean, self.var = None, None
        self.z2 = Lambda(self._sampling, name="latent2")(
            [self.z2_mean, self.z2_log_sigma])

        self.mean, self.var = tf.nn.moments(self.z2, axes=[0, 1])

        self.z1 = decoder2(self.z2)

        self.d1_mean, self.d1_log_sigma = Lambda(self._split)(d1)
        d1 = Lambda(self._sampling, name="latentD1")(
            [self.d1_mean, self.d1_log_sigma])

        self.z1 = Lambda(self._importance_weight)([self.z1, d1])

        reco = decoder1(self.z1)

        self.beta = beta

        model = Model(input_img, reco)
        model.compile(optimizer=RMSprop(lr=lr, decay=decay),
                      loss=self.bernoulli)

        self.latent_model = model
        self.latent_dim = self.latent_dim2

        return model, model

    def bernoulli(self, x_true, x_reco):
        return -tf.reduce_mean(tfd.Bernoulli(x_reco)._log_prob(x_true)) - self.beta * 0.5 * K.mean(
            1 + self.z2_log_sigma - K.square(self.z2_mean) - K.exp(self.z2_log_sigma), axis=-1)
