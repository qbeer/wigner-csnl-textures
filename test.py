from csnl import DataGenerator, DenseAutoEncoder, DenseVAE, LadderVAE, \
 ModelTrainer, VAEPlotter, SmallConvolutionalVAE, SmallDenseVAE
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = SmallConvolutionalVAE(input_shape=(70, 28, 28, 1), latent_dim=256)

model_trainer = ModelTrainer(conv_vae, data_gen, loss_fn="normal", beta=1, lr=1e-3)
model_trainer.fit(1, 1)

plotter = VAEPlotter(model_trainer, data_gen)
plotter.grid_plot()
