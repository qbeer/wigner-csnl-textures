from abc import abstractmethod
from keras.layers import Input, Dense, PReLU, Lambda, Add
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
        encoder = Model(input_tensor, x)
        return encoder

    def encoder2(self):
        input_tensor = Input(shape=(self.latent_dim1,))
        x = Dense(256)(input_tensor)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
        x = Dense(128)(x)
        x = PReLU()(x)
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
        latent1 = Input(shape=(self.latent_dim1,))
        x = Dense(256)(latent1)
        x = PReLU()(x)
        x = Dense(512)(x)
        x = PReLU()(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent1, reco)
        return decoder

    def _get_sigma(self, args):
        sigma1, sigma2 = args
        return K.pow(K.pow(sigma1, -2) + K.pow(sigma2, -2), -1)

    def _get_mean(self, args):
        mean1, sigma1, mean2, sigma2 = args
        return (mean1 * K.pow(sigma1, -2) + mean2 * K.pow(sigma2, -2)) * self._get_sigma([sigma1, sigma2])

    def _get_sigma_gen(self, args):
        sigma1 = args
        return K.pow(sigma1, 2)

    def _get_mean_gen(self, args):
        mean1, sigma1 = args
        return mean1 * K.pow(sigma1, -2) * self._get_sigma_gen(sigma1)

    def _sample(self, args):
        print("sample")
        z_mean, z_sigma = args
        dist = tfd.Normal(loc=z_mean, scale=z_sigma)
        return dist.sample()

    def _reparametrize(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.BATCH_SIZE, self.latent_dim2), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_compiled_model(self, *args):
        _, lr, decay, self.observation_noise, self.beta = args
        input_img = Input(batch_shape=self.input_shape)

        encoder1 = self.encoder1()
        encoder2 = self.encoder2()
        decoder1 = self.decoder1()
        decoder2 = self.decoder2()

        d1 = encoder1(input_img)
        d2 = encoder2(d1)

        # Reparametrization deepest layer
        self.z2_mean, self.z2_log_sigma = Dense(
            self.latent_dim2)(d2), Dense(self.latent_dim2)(d2)

        self.z2 = Lambda(self._reparametrize, name="latent")(
            [self.z2_mean, self.z2_log_sigma])

        # Top down and bottom up mean and variance calculation
        self.z1_intermediate = decoder2(self.z2)

        self.z1_mean_TD, self.z1_sigma_TD = Dense(self.latent_dim1)(
            self.z1_intermediate), Dense(self.latent_dim1)(self.z1_intermediate)

        self.z1_mean_BU, self.z1_sigma_BU = Dense(
            self.latent_dim1)(d1), Dense(self.latent_dim1)(d1)

        # Combine mean and sigma
        self.z1_sigma = Lambda(self._get_sigma)(
            [self.z1_sigma_BU, self.z1_sigma_TD])

        self.z1_mean = Lambda(self._get_mean)(
            [self.z1_mean_TD, self.z1_sigma_TD, self.z1_mean_BU, self.z1_sigma_BU])

        # Samlpe z1!
        self.z1 = Lambda(self._sample, name="sampling_z1")(
            [self.z1_mean, self.z1_sigma])

        reco = decoder1(self.z1)

        print("Reco shape : ", reco.get_shape())

        model = Model(input_img, reco)
        model.compile(optimizer=RMSprop(lr=lr, decay=decay),
                      loss=self.bernoulli, metrics=[self.KL_divergence])

        # Generative model
        latent_input = Input(shape=(self.latent_dim2,))
        gen2 = decoder2(latent_input)
        gen_reco = decoder1(gen2)
        generative_model = Model(latent_input, gen_reco)

        # Model for latent inference
        z2 = self.z2
        self.latent_model = Model(input_img, outputs=[reco, z2])

        self.latent_dim = self.latent_dim2

        return model, generative_model

    def bernoulli(self, x_true, x_reco):
        return -tf.reduce_mean(tfd.Bernoulli(x_reco)._log_prob(x_true)) \
            + self.KL_divergence(None, None) #+ self.KL_divergence1(None, None)

    def KL_divergence(self, x, y):
        return - self.beta * 0.5 * K.mean(
            1 + self.z2_log_sigma - K.square(self.z2_mean) - K.exp(self.z2_log_sigma), axis=-1)

    def KL_divergence1(self, x, y):
        mean, sigma = tf.nn.moments(self.z1, axes=[0])
        return - self.beta * K.mean(K.log(self.z1_sigma) / K.log(sigma)\
             + (K.pow(sigma, 2) + K.pow(mean - self.z1_mean, 2)/(2 * K.pow(sigma, 2)) - 0.5), axis=-1)