import os
from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLadderVAE, VAEPlotter, ModelTrainer

import tensorflow as tf
import tensorflow_probability as tfp
import keras as K

print(tf.__version__, tfp.__version__, K.__version__)

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=70,
                                          file_path=os.getcwd() +
                                          '/csnl/data/scramtex_700_28px.pkl',
                                          whiten=False)

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=70,
                         file_path=os.getcwd() +
                         '/csnl/data/scramtex_700_28px.pkl',
                         whiten=False)

LATENT_DIM1 = 16 * 4
LATENT_DIM2 = 16

vae = DenseLadderVAE(input_shape=(70, 28 * 28),
                     latent_dim1=LATENT_DIM1,
                     latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=5e-4,
                       decay=1e-4,
                       beta=1)

trainer.fit(10, 2000, contrast=True, warm_up=True, make_gif=True)

plotter = VAEPlotter(trainer, data_gen, data_gen_labels, grid_size=7)
plotter.plot_contrast_correlations()
plotter.plot_label_correlations()
plotter.grid_plot()
plotter.generate_samples()
