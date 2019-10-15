from .ladder_vae import LadderVAE
from keras.layers import Input, Dense, ReLU, Conv2D, PReLU, Flatten, UpSampling2D, Reshape
from keras.models import Model


class DenseConvLadderVAE(LadderVAE):
    def encoder1(self):
        input_img = Input(shape=self.input_shape[1:])
        x = Conv2D(512, (2, 2), strides=(2, 2), padding='valid')(input_img)
        x = PReLU()(x)
        x = Conv2D(256, (2, 2), strides=(2, 2), padding='valid')(x)
        x = PReLU()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.latent_dim1)(x)
        encoder = Model(input_img, x, name="conv_encoder_1")
        return encoder

    def encoder2(self):
        input_tensor = Input(shape=(self.latent_dim1, ))
        x = Dense(512)(input_tensor)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        encoder = Model(input_tensor, x, name="dense_encoder_2")
        return encoder

    def decoder2(self):
        latent2 = Input(shape=(self.latent_dim2, ))
        x = Dense(256)(latent2)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        x = Dense(self._mean_sigma_input_shape)(x)
        reco = ReLU()(x)
        decoder = Model(latent2, reco, name="dense_decoder_2")
        return decoder

    def decoder1(self):
        latent1 = Input(shape=(self.latent_dim1, ))
        x = Dense(1296, activation='relu')(latent1)
        x = Reshape(target_shape=(9, 9, 16))(x)
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
        decoder = Model(latent1, reco, name="conv_decoder_1")
        return decoder
