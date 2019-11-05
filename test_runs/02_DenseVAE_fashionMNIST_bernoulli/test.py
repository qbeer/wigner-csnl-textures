import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseVAE, VAEPlotter, ModelTrainer

from shutil import copyfile

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=100,
                                          file_path=os.getcwd() +
                                          '/csnl/data/fashion_mnist.pkl',
                                          binarize=True)

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         '/csnl/data/fashion_mnist.pkl',
                         binarize=True)

LATENT_DIM1 = 32
LATENT_DIM2 = 4

vae = DenseVAE(
    input_shape=(100, 28 * 28),
    #          latent_dim1=LATENT_DIM1,
    latent_dim=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="bernoulli",
                       lr=1e-4,
                       decay=1e-7,
                       beta=1)

trainer.fit(100, 200, contrast=False, warm_up=False, make_gif=False)

plotter = VAEPlotter(trainer, data_gen, data_gen_labels, grid_size=8)
plotter.plot_contrast_correlations()
plotter.plot_label_correlations()
plotter.grid_plot()
plotter.generate_samples()
#plotter.plot_td_bu_values(LATENT_DIM1)

copyfile(os.getcwd() + "/test.py", os.getcwd() + "/results/test.py")
