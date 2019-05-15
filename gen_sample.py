import torch
from pro_gan_pytorch import PRO_GAN as pg
import numpy as np
from torchvision import transforms
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="Script to generate samples from ProGAN generator")
parser.add_argument("--alpha", type=float, default=0.5, help="value of alpha for fade-in effect")
parser.add_argument("--depth", type=int, default=6, help="current depth from where output is required")
parser.add_argument("--cuda", type=int, default=1, help="Use GPU (1) or CPU (0)")
parser.add_argument("--out", type=str, default="./generated/0.jpg", help="directory to dump the generated image")
parset.add_argument("--generator", type=str, default="models/GAN_GEN_SHADOW_6.pth", help="path to generator state_dict")
opt = parser.parse_args()

device = torch.device("cuda" if opt.cuda else "cpu")

gen = torch.nn.DataParallel(pg.Generator(depth=7, latent_size=256))
gen.load_state_dict(torch.load(opt.generator, map_location=str(device)))

z = torch.randn(1, 256).float()
z = z.cuda() if opt.cuda else z

out = gen(z, opt.depth, opt.alpha).cpu().detach().squeeze(0)

save_image(out, opt.out)