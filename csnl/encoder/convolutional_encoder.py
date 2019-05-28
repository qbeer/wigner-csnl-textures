from .encoder import Encoder
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, PReLU, Dense, Flatten, Reshape
from keras.models import Model


class ConvolutionalEncoder(Encoder):

    def _encoder(self):
        input_img = Input(shape=self.input_shape[1:])
        x = Conv2D(512, (2, 2), padding='same')(input_img)
        x = PReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(256, (2, 2), padding='same')(x)
        x = PReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = PReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        encoded = Flatten()(x)
        encoder = Model(input_img, encoded)
        return encoder

    def _decoder(self):
        latent = Input(shape=(self.latent_dim,))
        x = Reshape((4, 4, self.latent_dim // 16))(latent)
        x = Conv2D(256, (2, 2), padding='same')(x)
        x = PReLU()(x)
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = PReLU()(x)
        x = Conv2D(512, (2, 2))(x)
        x = PReLU()(x)
        x = Conv2D(1024, (2, 2), padding='same')(x)
        x = PReLU()(x)
        x = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(x)
        x = PReLU()(x)
        reco = Conv2DTranspose(self.input_shape[-1], (2, 2), strides=(2, 2))(x)
        decoder = Model(latent, reco)
        return decoder
