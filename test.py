import os
from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseAutoEncoder, ConvolutionalAutoEncoder, VAEPlotter, ModelTrainer

from shutil import copyfile

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=100,
                                          file_path=os.getcwd() +
                                          '/csnl/data/textures_42000_28px.pkl',
                                          contrast_normalize=True)

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         '/csnl/data/textures_42000_28px.pkl',
                         contrast_normalize=True)

LATENT_DIM1 = 16 * 4
LATENT_DIM2 = 16

vae = ConvolutionalAutoEncoder(
    input_shape=(100, 28, 28, 1),
    #latent_dim1=LATENT_DIM1,
    latent_dim=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normalDiag",
                       lr=5e-4,
                       decay=1e-4,
                       beta=None)

trainer.fit(100, 500, contrast=False, warm_up=False, make_gif=False)

plotter = VAEPlotter(trainer, data_gen, data_gen_labels, grid_size=8)
plotter.plot_contrast_correlations(LATENT_DIM1)
plotter.plot_label_correlations()
plotter.grid_plot()
plotter.generate_samples()
plotter.plot_td_bu_values(LATENT_DIM1)

copyfile(os.getcwd() + "/test.py", os.getcwd() + "/results/test.py")
