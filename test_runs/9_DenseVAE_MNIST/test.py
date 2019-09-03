import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseVAE, VAEPlotter, ModelTrainer

data_gen_labels = DataGeneratorWithLabels(image_shape=(28, 28, 1),
                                          batch_size=200,
                                          file_path=os.getcwd() +
                                          "/csnl/data/mnist.pkl",
                                          whiten=False,
                                          contrast_normalize=False)

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=200,
                         file_path=os.getcwd() +
                         "/csnl/data/mnist.pkl",
                         whiten=False,
                         contrast_normalize=False)

LATENT_DIM = 2

vae = DenseVAE(input_shape=(200, 28 * 28), latent_dim=LATENT_DIM)

trainer = ModelTrainer(vae, data_gen, loss_fn="normal", lr=1e-5, decay=1e-5, beta=1)

trainer.fit(100, 500, warm_up=True)

plotter = VAEPlotter(trainer, data_gen, data_gen_labels, grid_size=16)
plotter.plot_contrast_correlations()
plotter.plot_label_correlations()
plotter.grid_plot()
plotter.generate_samples()

copyfile(os.getcwd() + "/test.py", os.getcwd() + "/results/test.py")