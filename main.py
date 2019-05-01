from csnl_util import DataGenerator
from autoencoders import DenseAutoEncoder, ConvolutionalAutoEncoder
from train import ModelTrainer
from variational_autoencoders import DenseVAE, ConvolutionalVAE

datagen = DataGenerator(image_shape=(28, 28, 1), batch_size=70)
dense_ae = DenseAutoEncoder((784,), 1024)

trainer = ModelTrainer(dense_ae, datagen, "normal")
#trainer.fit(10, 5)

conv_ae = ConvolutionalAutoEncoder(input_shape=(28, 28, 1), latent_dim=16*2)

conv_trainer = ModelTrainer(conv_ae, datagen, "normal")
#conv_trainer.fit(5, 3)

dense_vae = ConvolutionalVAE(input_shape=(28, 28, 1), latent_dim=16*1, BATCH_SIZE=70)

dense_vae_trainer = ModelTrainer(dense_vae, datagen, "normal")
dense_vae_trainer.fit(1, 2)

from visualize import VAEPlotter

plotter = VAEPlotter(dense_vae_trainer.model, dense_vae_trainer.generator, datagen)

plotter.grid_plot()