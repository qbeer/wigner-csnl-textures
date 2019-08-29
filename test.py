import os
from csnl import DataGeneratorWithLabels, DataGenerator, \
<<<<<<< HEAD
    SmallDenseLadderVAE, VAEPlotter, ModelTrainer
=======
    SmallDenseLinearLadderVAE, VAEPlotter, ModelTrainer
>>>>>>> a26c7fd8881dc340ef8c4cfc4567e455fc6ccb7e

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=100,
                                          file_path=os.getcwd() +
                                          '/csnl/data/textures_42000_28px.pkl',
                                          whiten=False, contrast_normalize=True)

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         '/csnl/data/textures_42000_28px.pkl',
                         whiten=False, contrast_normalize=True)

LATENT_DIM1 = 16 * 8
LATENT_DIM2 = 16 * 2

<<<<<<< HEAD
vae = SmallDenseLadderVAE(input_shape=(100, 28 * 28),
=======
vae = SmallDenseLinearLadderVAE(input_shape=(100, 28 * 28),
>>>>>>> a26c7fd8881dc340ef8c4cfc4567e455fc6ccb7e
                     latent_dim1=LATENT_DIM1,
                     latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=5e-5,
                       decay=1e-5,
                       beta=1)

trainer.fit(250, 1000, contrast=True, warm_up=True, make_gif=True)

plotter = VAEPlotter(trainer, data_gen, data_gen_labels, grid_size=8)
plotter.plot_contrast_correlations(latent_dim2=LATENT_DIM1)
plotter.plot_label_correlations()
plotter.grid_plot()
plotter.generate_samples()
