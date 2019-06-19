from csnl import DenseLadderVAE, ModelTrainer, VAEPlotter, DataGenerator
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = DenseLadderVAE(input_shape=(70, 28*28),
                          latent_dim1=128, latent_dim2=64)

model_trainer = ModelTrainer(
    conv_vae, data_gen, loss_fn="normal", beta=1, lr=5e-6)
model_trainer.fit(10, 50)

plotter = VAEPlotter(model_trainer, data_gen)
plotter.grid_plot()
plotter.generate_samples()
