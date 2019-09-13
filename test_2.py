import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLadderVAE_BN, VAEPlotter, ModelTrainer

data_gen = DataGenerator(image_shape=(28, 28, 1), batch_size=70)

LATENT_DIM2 = 16
LATENT_DIM1 = 64

vae = DenseLadderVAE_BN(input_shape=(70, 28 * 28),
                        latent_dim2=LATENT_DIM2,
                        latent_dim1=LATENT_DIM1)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=5e-4,
                       decay=1e-5,
                       beta=1)

trainer.fit(100, 1000, warm_up=True)

plotter = VAEPlotter(trainer, data_gen, None, grid_size=10)
plotter.grid_plot()
plotter.generate_samples()

copyfile(os.getcwd() + "/test_2.py", os.getcwd() + "/results/test_2.py")