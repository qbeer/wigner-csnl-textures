from .ladder_vae_bn import LadderVAE_BN
from keras.layers import Input, Conv2D, ReLU, Dense, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras.models import Model
import numpy as np


class ConvLadderVAE_BN(LadderVAE_BN):
    def encoder1(self):
        input_img = Input(shape=self.input_shape[1:])
        x = Conv2D(256, (2, 2), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        encoded = Dense(self.latent_dim1)(x)
        encoder = Model(input_img, encoded, name="convolutional_encoder_1")
        return encoder

    def encoder2(self):
        input_img = Input(shape=(self.latent_dim1, ))
        x = Dense(256)(input_img)
        x = BatchNormalization()(x)
        x = Reshape(target_shape=(16, 16, 1))(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        encoded = Dense(self.latent_dim2)(x)
        encoder = Model(input_img, encoded, name="convolutional_encoder_2")
        return encoder

    def decoder2(self):
        latent2 = Input(shape=(self.latent_dim2, ))
        x = Dense(256)(latent2)
        x = BatchNormalization()(x)
        x = Reshape(target_shape=(16, 16, 1))(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        x = Dense(self._mean_sigma_input_shape)(x)
        x = BatchNormalization()(x)
        reco = ReLU()(x)
        decoder = Model(latent2, reco, name="convolutional_decoder_2")
        return decoder

    def decoder1(self):
        latent1 = Input(shape=(self.latent_dim1, ))
        x = Dense(1024)(latent1)
        x = BatchNormalization()(x)
        x = Reshape(target_shape=(32, 32, 1))(x)
        x = Conv2D(256, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        x = Dense(np.prod(self.input_shape[1:]))(x)
        reco = Reshape(target_shape=self.input_shape[1:])(x)
        decoder = Model(latent1, reco, name="convolutional_decoder_1")
        return decoder
