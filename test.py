from csnl import DataGenerator, DenseAutoEncoder, ModelTrainer, VAEPlotter, SmallConvolutionalVAE, SmallDenseVAE
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=35,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = SmallDenseVAE(input_shape=(35, 28*28), latent_dim=16)

model_trainer = ModelTrainer(conv_vae, data_gen, loss_fn="normal", beta=1)
model_trainer.fit(2, 100, contrast=True)

plotter = VAEPlotter(model_trainer, data_gen)
plotter.grid_plot()
