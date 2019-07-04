from csnl import SmallDenseLadderVAE_BN, SmallDenseLadderVAE,\
    DenseLadderVAE, ModelTrainer, VAEPlotter, DataGenerator, GifCallBack, SmallDenseVAE
import os
from keras.utils import plot_model

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl', whiten=True)
conv_vae = SmallDenseVAE(input_shape=(70, 28*28),
                         latent_dim=256)

model_trainer = ModelTrainer(
    conv_vae, data_gen, loss_fn="normal", beta=1, lr=5e-6)
plot_model(model_trainer.model, to_file='vae.png')
model_trainer.fit(5, 50, contrast=True)
