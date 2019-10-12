import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLinearLadderVAE, VAEPlotter, ModelTrainer

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         "/csnl/data/textures_42000_28px.pkl")

data_gen_label = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                         batch_size=100,
                                         file_path=os.getcwd() +
                                         "/csnl/data/textures_42000_28px.pkl")

LATENT_DIM2 = 16
LATENT_DIM1 = 16 * 4

vae = DenseLinearLadderVAE(input_shape=(100, 28 * 28),
                           latent_dim1=LATENT_DIM1,
                           latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=1e-5,
                       decay=1e-5,
                       beta=100)

trainer.fit(1500, 750, warm_up=True)

plotter = VAEPlotter(trainer, data_gen, data_gen_label, grid_size=10)

plotter.grid_plot()
plotter.generate_samples()
plotter.plot_contrast_correlations(LATENT_DIM1)
plotter.plot_label_correlations()

copyfile(os.getcwd() + "/test.py", os.getcwd() + "/results/test.py")