import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLadderVAE, VAEPlotter, ModelTrainer

data_gen = DataGenerator(image_shape=(28, 28, 3),
                         batch_size=100,
                         file_path=os.getcwd() + "/csnl/data/cifar10.pkl")

LATENT_DIM2 = 16
LATENT_DIM1 = 16 * 4

vae = DenseLadderVAE(input_shape=(100, 28 * 28 * 3),
                     latent_dim1=LATENT_DIM1,
                     latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=1e-5,
                       decay=1e-5,
                       beta=100)

trainer.fit(2000, 1000, warm_up=True)

plotter = VAEPlotter(trainer, data_gen, data_gen, grid_size=10)

plotter.grid_plot()
plotter.generate_samples()

copyfile(os.getcwd() + "/test.py", os.getcwd() + "/results/test.py")