from .variational_autoencoder import VariationalAutoEncoder
from keras.layers import Input, Conv2D, ReLU, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model


class SmallConvolutionalVAE(VariationalAutoEncoder):
    def _encoder(self):
        input_img = Input(shape=self.input_shape[1:])
        x = Conv2D(256, (2, 2), padding='same')(input_img)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        encoded = Flatten()(x)
        encoder_model = Model(input_img, encoded)
        return encoder_model

    def _decoder(self):
        latent = Input(shape=(self.latent_dim,))
        x = Reshape((4, 4, self.latent_dim // 16))(latent)
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = ReLU()(x)
        x = Conv2D(128, (2, 2))(x)
        x = ReLU()(x)
        x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
        x = ReLU()(x)
        reco = Conv2DTranspose(
            self.input_shape[-1], (2, 2), strides=(2, 2), padding='same')(x)
        decoder_model = Model(latent, reco)
        return decoder_model
