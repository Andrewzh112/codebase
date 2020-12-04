import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import torchvision

from vae.model import VAE
from vae.main import parser, args


class Sampler:
    def __init__(self, sample_path='vae/samples', ext='.jpg'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.vae = torch.nn.DataParallel(VAE(args), device_ids=args.device_ids).to(self.device)
        self.vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/VAE.pth")['model'])
        self.vae.eval()
        Path(sample_path).mkdir(parents=True, exist_ok=True)

        self.sample_path = sample_path
        self.ext = ext

    def sample(self):
        with torch.no_grad():
            samples = self.vae.module.sample(num_samples=args.sample_size)
        torchvision.utils.save_image(
            samples,
            self.sample_path + f'/sample_{int(datetime.now().timestamp()*1e6)}' + self.ext)

    def generate_walk_z(self):
        z = torch.randn(args.z_dim, device=self.device)
        z = z.repeat(args.sample_size).view(args.sample_size, args.z_dim)
        walk_dim = np.random.choice(list(range(args.z_dim)))
        z[:, walk_dim] = torch.linspace(-2, 2, args.sample_size)
        return z

    def walk(self):
        z = self.generate_walk_z()
        with torch.no_grad():
            samples = self.vae.module.sample(z=z)
        torchvision.utils.save_image(
            samples,
            self.sample_path + f'/walk_{int(datetime.now().timestamp()*1e6)}' + self.ext)


if __name__ == '__main__':
    sampler = Sampler()
    if args.sample:
        sampler.sample()
    if args.walk:
        sampler.walk()
