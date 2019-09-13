import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLadderVAE_BN, VAEPlotter, ModelTrainer

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         "/csnl/data/textures_42000_28px.pkl",
                         contrast_normalize=True)

data_gen_label = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                         batch_size=100,
                                         file_path=os.getcwd() +
                                         "/csnl/data/textures_42000_28px.pkl",
                                         contrast_normalize=True)

LATENT_DIM2 = 16
LATENT_DIM1 = 64

vae = DenseLadderVAE_BN(input_shape=(100, 28 * 28),
                        latent_dim2=LATENT_DIM2,
                        latent_dim1=LATENT_DIM1)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=5e-4,
                       decay=1e-5,
                       beta=1)

trainer.fit(1000, 1000, warm_up=True)

plotter = VAEPlotter(trainer, data_gen, data_gen_label, grid_size=10)
plotter.grid_plot()
plotter.generate_samples()
plotter.plot_contrast_correlations(LATENT_DIM1)
plotter.plot_label_correlations()

copyfile(os.getcwd() + "/test_2.py", os.getcwd() + "/results/test.py")