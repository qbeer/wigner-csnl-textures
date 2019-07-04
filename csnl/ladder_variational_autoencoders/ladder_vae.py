from ..losses import Losses
from abc import abstractmethod
from keras.layers import Input, Dense, ReLU, Lambda, Add
from keras.models import Model
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class LadderVAE:
    def __init__(self, input_shape, latent_dim1, latent_dim2, mean_variance_input_shape=256):
        self.input_shape = input_shape
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.BATCH_SIZE = self.input_shape[0]
        self._mean_variance_input_shape = mean_variance_input_shape  # arbitrary

    @abstractmethod
    def encoder1(self):
        pass

    @abstractmethod
    def encoder2(self):
        pass

    @abstractmethod
    def decoder2(self):
        pass

    @abstractmethod
    def decoder1(self):
        pass

    def mean_log_variance_model(self):
        inp = Input(shape=(self._mean_variance_input_shape,))
        mean, log_var = Dense(self.latent_dim1)(
            inp), Dense(self.latent_dim1)(inp)
        model = Model(inp, [mean, log_var], name="mean_log_variance_model")
        return model

    def _get_sigma(self, args):
        log_sigma1, log_sigma2 = args
        return K.pow(K.pow(K.exp(log_sigma1) + 1e-12, -2) +
                     K.pow(K.exp(log_sigma2) + 1e-12, -2) + 1e-12, -1)

    def _get_mean(self, args):
        mean1, log_sigma1, mean2, log_sigma2 = args
        return (mean1 * K.pow(K.exp(log_sigma1) + 1e-12, -2) +
                mean2 * K.pow(K.exp(log_sigma2) + 1e-12, -2)) * self._get_sigma([log_sigma1, log_sigma2])

    def _get_sigma_gen(self, args):
        log_sigma1 = args
        return K.pow(K.exp(log_sigma1), 2)

    def _get_mean_gen(self, args):
        mean1, log_sigma1 = args
        return mean1 * K.pow(K.exp(log_sigma1) + 1e-12, -2) * self._get_sigma_gen(log_sigma1)

    def _sample(self, args):
        z_mean, z_sigma = args
        dist = tfd.Normal(loc=z_mean, scale=z_sigma)
        return dist.sample()

    def _reparametrize(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.BATCH_SIZE, self.latent_dim2), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def get_compiled_model(self, *args):
        loss_fn, lr, decay, self.observation_noise, self.beta = args
        input_img = Input(batch_shape=self.input_shape)

        encoder1 = self.encoder1()
        encoder2 = self.encoder2()
        decoder1 = self.decoder1()
        decoder2 = self.decoder2()
        mean_log_var_model_for_top_down_calc = self.mean_log_variance_model()

        d1 = encoder1(input_img)
        d2 = encoder2(d1)

        # Reparametrization deepest layer
        self.z2_mean, self.z2_log_sigma = Dense(
            self.latent_dim2, name="mean_z2")(d2), Dense(self.latent_dim2, name="log_sigma_z2")(d2)

        self.z2 = Lambda(self._reparametrize, name="sampling_z2")(
            [self.z2_mean, self.z2_log_sigma])

        # Top down and bottom up mean and variance calculation
        self.z1_intermediate = decoder2(self.z2)

        self.z1_mean_TD, self.z1_log_sigma_TD = mean_log_var_model_for_top_down_calc(
            self.z1_intermediate)

        self.z1_mean_BU, self.z1_log_sigma_BU = Dense(
            self.latent_dim1, name="bottom_up_mean")(d1), Dense(self.latent_dim1, name="bottom_up_log_sigma")(d1)

        # Combine mean and sigma
        self.z1_sigma = Lambda(self._get_sigma, name="calculate_sigma_z1")(
            [self.z1_log_sigma_BU, self.z1_log_sigma_TD])

        self.z1_mean = Lambda(self._get_mean, name="calculate_mean_z1")(
            [self.z1_mean_TD, self.z1_log_sigma_TD, self.z1_mean_BU, self.z1_log_sigma_BU])

        # Samlpe z1!
        self.z1 = Lambda(self._sample, name="sampling_z1")(
            [self.z1_mean, self.z1_sigma])

        reco = decoder1(self.z1)

        losses = Losses(loss_fn, self.observation_noise, self.beta,
                        z2_mean=self.z2_mean, z2_log_sigma=self.z2_log_sigma,
                        z_mean=self.z1_mean, z_sigma=self.z1_sigma,
                        z_mean_TD=self.z1_mean_TD, z_log_sigma_TD=self.z1_log_sigma_TD)

        model = Model(input_img, reco)
        model.compile(optimizer=RMSprop(lr=lr, decay=decay),
                      loss=losses.loss, metrics=[losses.KL_divergence])

        # Generative model
        latent_input = Input(shape=(self.latent_dim2,))
        gen2 = decoder2(latent_input)

        # Using same TD mean var generator as before
        gen_mean, gen_log_sigma = mean_log_var_model_for_top_down_calc(gen2)

        # Combine mean and sigma for generative model
        gen_sigma = Lambda(self._get_sigma_gen)(
            [gen_log_sigma])

        gen_mean = Lambda(self._get_mean_gen)(
            [gen_mean, gen_log_sigma])

        # Samlping
        gen2 = Lambda(self._sample)(
            [gen_mean, gen_sigma])

        gen_reco = decoder1(gen2)
        generative_model = Model(latent_input, gen_reco)

        # Model for latent inference
        z2 = self.z2
        latent_model = Model(input_img, outputs=[reco, z2])

        self.latent_dim = self.latent_dim2

        return model, generative_model, latent_model
