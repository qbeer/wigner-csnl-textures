import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from shutil import copyfile

from csnl import DataGeneratorWithLabels, DataGenerator, \
    DenseLadderVAE, VAEPlotter, ModelTrainer

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=100,
                         file_path=os.getcwd() +
                         "/csnl/data/natural_80000_28px.pkl")

LATENT_DIM2 = 16
LATENT_DIM1 = 16 * 4

vae = DenseLadderVAE(input_shape=(100, 28 * 28),
                     latent_dim1=LATENT_DIM1,
                     latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae,
                       data_gen,
                       loss_fn="normal",
                       lr=1e-5,
                       decay=1e-5,
                       beta=100)

trainer.fit(1500, 1000, warm_up=True)

plotter = VAEPlotter(trainer, data_gen, None, grid_size=10)
plotter.grid_plot()
plotter.generate_samples()

copyfile(os.getcwd() + "/test.py", os.getcwd() + "/results/test.py")