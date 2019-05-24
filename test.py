from csnl import DataGenerator, DenseAutoEncoder, LadderVAE, \
 ModelTrainer, VAEPlotter, SmallConvolutionalVAE, SmallDenseVAE
import os

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=35,
                         file_path=os.getcwd() + '/csnl/data/scramtex_700_28px.pkl')
conv_vae = LadderVAE(input_shape=(35, 28*28), latent_dim1=64, latent_dim2=32)

model_trainer = ModelTrainer(conv_vae, data_gen, loss_fn="normal", beta=1, lr=1e-6)
model_trainer.fit(10, 200)

plotter = VAEPlotter(model_trainer, data_gen)
plotter.grid_plot()
