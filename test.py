from csnl import ConvLadderVAE, ModelTrainer, VAEPlotter, DataGenerator, DenseLadderVAE
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = ConvLadderVAE(input_shape=(70, 28, 28, 1), latent_dim1=128, latent_dim2=64)

model_trainer = ModelTrainer(conv_vae, data_gen, loss_fn="normal", beta=1, lr=1e-3)
model_trainer.fit(5, 100)

plotter = VAEPlotter(model_trainer, data_gen)
plotter.grid_plot()
plotter.generate_samples()
