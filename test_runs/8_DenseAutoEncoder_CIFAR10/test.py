import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseAutoEncoder, VAEPlotter, ModelTrainer

data_gen = DataGenerator(image_shape=(28, 28, 3),
                         batch_size=100,
                         whiten=False,
                         file_path=os.getcwd() + "/csnl/data/cifar10.pkl")

LATENT_DIM = 32

vae = DenseAutoEncoder(input_shape=(100, 28 * 28 * 3), latent_dim=LATENT_DIM)

trainer = ModelTrainer(vae, data_gen, loss_fn="normal", lr=5e-4, decay=1e-5)

trainer.fit(150, 1000)

plotter = VAEPlotter(trainer, data_gen, None, grid_size=16)
plotter.grid_plot()
plotter.generate_samples()

copyfile(os.getcwd() + "test_2.py", os.getcwd() + "/results/test.py")