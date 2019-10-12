import os
from csnl import DataGenerator, DenseLadderVAE, ModelTrainer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_gen = DataGenerator(image_shape=(28, 28, 1),
                         batch_size=70,
                         file_path=os.getcwd() +
                         '/csnl/data/textures_42000_28px.pkl',
                         whiten=False,
                         contrast_normalize=True)

LATENT_DIM1 = 16 * 4
LATENT_DIM2 = 16

vae = DenseLadderVAE(input_shape=(70, 28 * 28),
                     latent_dim1=LATENT_DIM1,
                     latent_dim2=LATENT_DIM2)

trainer = ModelTrainer(vae, data_gen, loss_fn='normal', beta=1)

from keras.utils import plot_model

plot_model(trainer.model, to_file='dense_lvae_keras.png')