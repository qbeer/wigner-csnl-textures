from csnl import SmallDenseLadderVAE_BN, SmallDenseLadderVAE,\
    DenseLadderVAE, ModelTrainer, VAEPlotter, DataGenerator, GifCallBack
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = SmallDenseLadderVAE(input_shape=(70, 28*28),
                               latent_dim1=256, latent_dim2=128)

model_trainer = ModelTrainer(
    conv_vae, data_gen, loss_fn="normal", beta=1, lr=5e-6)
model_trainer.fit(20, 200)
