from csnl import DataGenerator, DenseVAE, ModelTrainer, VAEPlotter
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70, file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = DenseVAE(input_shape=(784,), latent_dim=16*4, BATCH_SIZE=70)

model_trainer = ModelTrainer(conv_vae, data_gen, loss_fn="normal")
model_trainer.fit(1, 1)

plotter = VAEPlotter(model_trainer.model, model_trainer.generator, data_gen)
plotter.grid_plot()