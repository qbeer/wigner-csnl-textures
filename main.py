from csnl_util import DataGenerator
from autoencoders import DenseAutoEncoder, ConvolutionalAutoEncoder
from train import ModelTrainer

datagen = DataGenerator(image_shape=(28, 28, 1), batch_size=70)
dense_ae = DenseAutoEncoder((784,), 1024)

trainer = ModelTrainer(dense_ae, datagen, "normal")
trainer.fit(10, 5)

conv_ae = ConvolutionalAutoEncoder(input_shape=(28, 28, 1), latent_dim=16*2)

conv_trainer = ModelTrainer(conv_ae, datagen, "normal")
conv_trainer.fit(5, 3)