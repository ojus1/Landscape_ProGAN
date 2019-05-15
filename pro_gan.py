touimport numpy as np
import torch
from torch import nn
from DataHelper import LandscapeImages, img_shape
from torchvision.utils import save_image
import pro_gan_pytorch.PRO_GAN as pg


def train_model(device_to_run):
    #Data
    dataset = LandscapeImages()

    #Hyperparameters
    depth = 7
    batch_sizes = [5, 5, 5, 5, 5, 5, 5]
    num_epochs = [10, 10, 15, 15, 15, 20, 25]
    fade_ins = [50, 50, 50, 50, 50, 50, 50]
    latent_size = 256
    
    gan = pg.ProGAN(device=device_to_run, latent_size=latent_size, depth=depth)

    gan.train(dataset=dataset, epochs=num_epochs, batch_sizes=batch_sizes, fade_in_percentage=fade_ins, num_workers=4)

if __name__ == "__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(device)
