import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.utils import plot_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLinLinLadderVAE, DenseLadderVAE,ConvLadderVAE,\
    VAEPlotter, ModelTrainer, DenseVAE, ConvolutionalVAE, DenseAutoEncoder

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         '/csnl/data/textures_42000_28px.pkl',
                         contrast_normalize=True)

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=100,
                                          file_path=os.getcwd() +
                                          '/csnl/data/textures_42000_28px.pkl',
                                          contrast_normalize=True)

LATENT_DIM1 = 16 * 8
LATENT_DIM2 = 8

vae = DenseAutoEncoder(input_shape=(100, 28*28),
                   # latent_dim1=LATENT_DIM1,
                    latent_dim=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=5e-4,
                       decay=1e-4,
                       beta=1)

plot_model(
    trainer.model,
    to_file='data/DenseAutoEncoder_vertical.png',
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    rankdir='TB',  # TB vertical, LR horizontal
    dpi=150)
