import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLinLinLadderVAE, VAEPlotter, ModelTrainer, DenseLadderVAE

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=500,
                         file_path=os.getcwd() +
                         '/csnl/data/textures_42000_28px.pkl',
                         contrast_normalize=True)

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=500,
                                          file_path=os.getcwd() +
                                          '/csnl/data/textures_42000_28px.pkl',
                                          contrast_normalize=True)

LATENT_DIM1 = 16 * 8
LATENT_DIM2 = 8

vae = DenseLadderVAE(input_shape=(500, 28 * 28),
                     latent_dim1=LATENT_DIM1,
                     latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=5e-4,
                       decay=1e-4,
                       beta=1)

trainer.model.load_weights('./results/model.h5')
trainer.generator.load_weights('./results/generator_model.h5')
trainer.latent_model.load_weights('./results/latent_model.h5')
plotter = VAEPlotter(trainer, data_gen, data_gen_labels, grid_size=8)
plotter.plot_contrast_correlations(LATENT_DIM1)
plotter.plot_label_correlations()
plotter.grid_plot()
plotter.generate_samples()
plotter.plot_td_bu_values(LATENT_DIM1, size=5)