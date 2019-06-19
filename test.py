from csnl import SmallDenseLadderVAE, DenseLadderVAE, ModelTrainer, VAEPlotter, DataGenerator, GifCallBack
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=100,
                         file_path=os.getcwd() + '/csnl/data/natural_24000_28px.pkl')
conv_vae = SmallDenseLadderVAE(input_shape=(100, 28*28),
                               latent_dim1=128, latent_dim2=64)

model_trainer = ModelTrainer(
    conv_vae, data_gen, loss_fn="normal", beta=1, lr=5e-6)
model_trainer.fit(3, 1)
