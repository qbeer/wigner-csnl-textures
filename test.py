from csnl import DataGenerator, DenseAutoEncoder, ModelTrainer, VAEPlotter, SmallConvolutionalVAE, SmallDenseVAE
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70, file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = SmallConvolutionalVAE(input_shape=(70, 28, 28, 1), latent_dim=16)

model_trainer = ModelTrainer(conv_vae, data_gen, loss_fn="normal")
model_trainer.fit(1, 1, contrast=True)

plotter = VAEPlotter(model_trainer, data_gen)
plotter.grid_plot()
plotter.generate_samples()
plotter.visualize_latent()