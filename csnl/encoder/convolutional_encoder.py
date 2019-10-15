from .encoder import Encoder
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, PReLU, Dense, Flatten, Reshape
from keras.models import Model


class ConvolutionalEncoder(Encoder):
    def _encoder(self):
        input_img = Input(shape=self.input_shape[1:])
        x = Conv2D(512, (2, 2), strides=(2, 2), padding='valid')(input_img)
        x = PReLU()(x)
        x = Conv2D(256, (2, 2), strides=(2, 2), padding='valid')(x)
        x = PReLU()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        encoder = Model(input_img, x, name="convolutional_encoder")
        return encoder

    def _decoder(self):
        latent = Input(shape=(self.latent_dim, ))
        x = Dense(243, activation='relu')(latent)
        x = Reshape((9, 9, 3))(x)
        x = Conv2D(128, (2, 2), padding='valid')(x)  # 8 x 8 x 128
        x = PReLU()(x)
        x = UpSampling2D(size=(2, 2),
                         interpolation='bilinear')(x)  # 16 x 16 x 128
        x = Conv2D(256, (2, 2), padding='valid')(x)  # 15 x 15 x 256
        x = PReLU()(x)
        x = UpSampling2D(size=(2, 2),
                         interpolation='bilinear')(x)  # 30 x 30 x 256
        x = Conv2D(512, (2, 2), padding='valid')(x)  # 29 x 29 x 512
        x = PReLU()(x)
        reco = Conv2D(self.input_shape[-1], (2, 2),
                      padding='valid')(x)  # 28 x 28 x input_channels
        decoder = Model(latent, reco, name="convolutional_decoder")
        return decoder
