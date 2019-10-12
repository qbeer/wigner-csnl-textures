from .ladder_vae_bn import LadderVAE_BN
from keras.layers import Input, Dense, ReLU, BatchNormalization
from keras.models import Model


class DenseLadderVAE_BN(LadderVAE_BN):
    def encoder1(self):
        input_tensor = Input(shape=self.input_shape[1:])
        x = Dense(1024)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(self.latent_dim1)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        encoder = Model(input_tensor, x, name="dense_encoder_1")
        return encoder

    def encoder2(self):
        input_tensor = Input(shape=(self.latent_dim1, ))
        x = Dense(512)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        encoder = Model(input_tensor, x, name="dense_encoder_2")
        return encoder

    def decoder2(self):
        latent2 = Input(shape=(self.latent_dim2, ))
        x = Dense(256)(latent2)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(self._mean_sigma_input_shape)(x)
        x = BatchNormalization()(x)
        reco = ReLU()(x)
        decoder = Model(latent2, reco, name="dense_decoder_2")
        return decoder

    def decoder1(self):
        latent1 = Input(shape=(self.latent_dim1, ))
        x = Dense(512)(latent1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(2048)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        reco = Dense(self.input_shape[1])(x)
        decoder = Model(latent1, reco, name="dense_decoder_1")
        return decoder
