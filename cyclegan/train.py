import torch
import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import itertools
from tqdm import tqdm
from data import CycleImageDataset
from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, make_images, get_random_ids


parser = argparse.ArgumentParser()
parser.add_argument('-train', action="store_true", default=False, help='Train cycleGAN models')
parser.add_argument('--dim_A', type=int, default=3, help='Numer of channels for class A')
parser.add_argument('--dim_B', type=int, default=3, help='Numer of channels for class B')
parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of ResNet Blocks for generators')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for model')
parser.add_argument('--lr_G', type=float, default=0.0002, help='Learning rate for generators')
parser.add_argument('--lr_D', type=float, default=0.0002, help='Learning rate for discriminators')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--starting_epoch', type=int, default=0, help='Starting epoch for resuming training')
parser.add_argument('--decay_epoch', type=int, default=100, help='Starting epoch when learning rate will start decay')
parser.add_argument('--load_shape', type=int, default=256, help='Initial image H or W')
parser.add_argument('--target_shape', type=int, default=224, help='Final image H or W')
parser.add_argument('--progress_interval', type=int, default=1, help='Save model and generated image every x epoch')
parser.add_argument('--sample_batches', type=int, default=25, help='How many generated images to sample')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--lambda_identity', type=float, default=0.1, help='Identity loss weight')
parser.add_argument('--lambda_cycle', type=float, default=10., help='Cycle loss weight')
parser.add_argument('--log_dir', type=str, default='logs', help='Path to where log files will be saved')
parser.add_argument('--save_img_dir', type=str, default='save_images', help='Path to where generated images will be saved')
parser.add_argument('--data_root', type=str, default='horse2zebra', help='Path to where image data is located')
parser.add_argument('--checkpoint_dir', type=str, default='model_weights', help='Path to where model weights will be saved')
args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(
            args.log_dir + f'/{int(datetime.now().timestamp()*1e6)}')
    G_AB = Generator(
            args.dim_A,
            args.dim_B,
            args.hidden_dim,
            args.n_res_blocks
        ).to(device)
    G_BA = Generator(
        args.dim_B,
        args.dim_A,
        args.hidden_dim,
        args.n_res_blocks
    ).to(device)
    D_A = Discriminator(args.dim_A, args.hidden_dim).to(device)
    D_B = Discriminator(args.dim_B, args.hidden_dim).to(device)

    dataset = CycleImageDataset(args.data_root, 'train', args.load_shape, args.target_shape)

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(),
        G_BA.parameters()),
        lr=args.lr_G,
        betas=args.betas
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(),
        lr=args.lr_D,
        betas=args.betas
    )
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(),
        lr=args.lr_D,
        betas=args.betas
    )

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A,
        lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B,
        lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
    )

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    pool_A = ReplayBuffer()
    pool_B = ReplayBuffer()

    pbar = tqdm(
            range(args.starting_epoch, args.n_epochs),
            total=(args.n_epochs - args.starting_epoch)
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in pbar:
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        disc_A_losses, disc_B_losses, gen_AB_losses = [], [], []
        real_As, real_Bs, fake_As, fake_Bs = [], [], [], []
        sampled_idx = get_random_ids(len(dataloader), args.sample_batches)
        for batch_idx, (real_A, real_B) in enumerate(dataloader):
            real_A = torch.nn.functional.interpolate(real_A, size=args.target_shape).to(device)
            real_B = torch.nn.functional.interpolate(real_B, size=args.target_shape).to(device)

            ### Update discriminator A ###
            with torch.no_grad():
                fake_A = G_BA(real_B)
                fake_A = pool_A.sample(fake_A)
            fake_logits = D_A(fake_A)
            disc_A_fake_loss = criterion_GAN(fake_logits, torch.zeros_like(fake_logits))
            real_logits = D_A(real_A)
            disc_A_real_loss = criterion_GAN(real_logits, torch.ones_like(real_logits))
            disc_A_loss = (disc_A_fake_loss + disc_A_real_loss) / 2
            disc_A_losses.append(disc_A_loss.item())

            ### Update discriminator B ###
            with torch.no_grad():
                fake_B = G_AB(real_A)
                fake_B = pool_B.sample(fake_B)
            fake_logits = D_B(fake_B)
            disc_B_fake_loss = criterion_GAN(fake_logits, torch.zeros_like(fake_logits))
            real_logits = D_B(real_B)
            disc_B_real_loss = criterion_GAN(real_logits, torch.ones_like(real_logits))
            disc_B_loss = (disc_B_fake_loss + disc_B_real_loss) / 2
            disc_B_losses.append(disc_B_loss.item())

            ### Update Generators ###
            ## Adversarial Loss ##
            fake_A = G_BA(real_B)
            fake_B = G_AB(real_A)
            fake_A_logits = D_A(fake_A)
            fake_B_logits = D_B(fake_B)
            adversarial_loss = criterion_GAN(fake_A_logits, torch.ones_like(fake_A_logits)) + criterion_GAN(fake_B_logits, torch.ones_like(fake_B_logits))

            ## cycle consistency loss ##
            cycle_A = G_BA(fake_B)
            cycle_B = G_AB(fake_A)
            cycle_loss = criterion_cycle(cycle_A, real_A) + criterion_cycle(cycle_B, real_B)
            
            # identity loss ##
            identity_A = G_BA(real_A)
            identity_B = G_AB(real_B)
            identity_loss = criterion_identity(identity_A, real_A) + criterion_identity(identity_B, real_B)

            # weighted generator loss
            gen_loss = adversarial_loss + args.lambda_identity*identity_loss + args.lambda_cycle*cycle_loss
            gen_AB_losses.append(gen_loss.item())

            disc_A_loss.backward(retain_graph=True)
            optimizer_D_A.step()
            disc_B_loss.backward(retain_graph=True)
            optimizer_D_B.step()
            gen_loss.backward()
            optimizer_G.step()
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()
            optimizer_G.zero_grad()

            if batch_idx in sampled_idx:
                real_As.append(real_A.detach().cpu())
                real_Bs.append(real_B.detach().cpu())
                fake_As.append(fake_A.detach().cpu())
                fake_Bs.append(fake_B.detach().cpu())

        images = [torch.cat(real_As), torch.cat(real_Bs), torch.cat(fake_As), torch.cat(fake_Bs)]
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        lr_scheduler_G.step()

        if (epoch + 1) % args.progress_interval == 0:
            writer.add_scalars('Train Losses', {
                    'Discriminator A': sum(disc_A_losses) / len(disc_A_losses),
                    'Discriminator B': sum(disc_B_losses) / len(disc_B_losses),
                    'Generator': sum(gen_AB_losses) / len(gen_AB_losses)
                }, global_step=epoch)
            writer.add_image('Fake A', make_images(fake_As), global_step=epoch)
            writer.add_image('Fake B', make_images(fake_Bs), global_step=epoch)
            writer.add_image('Real A', make_images(real_As), global_step=epoch)
            writer.add_image('Real B', make_images(real_Bs), global_step=epoch)
            if (epoch + 1) % 10 == 0:
                if not os.path.isdir(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                torch.save({
                    'G_AB': G_AB.state_dict(),
                    'G_BA': G_BA.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'D_A': D_A.state_dict(),
                    'optimizer_D_A': optimizer_D_A.state_dict(),
                    'D_B': D_B.state_dict(),
                    'optimizer_D_B': optimizer_D_B.state_dict()
                }, f"{args.checkpoint_dir}/cycleGAN_{epoch}.pth")

                # saving space, only saving latest weights
                if epoch > 10:
                    os.remove(f"{args.checkpoint_dir}/cycleGAN_{epoch - 10}.pth")
        tqdm.write('#########################################################')
        tqdm.write(
            f'Epoch {epoch + 1}/{args.n_epochs}, \
                Train Disc A loss: {sum(disc_A_losses) / len(disc_A_losses):.3f}, \
                Train Disc B loss: {sum(disc_B_losses) / len(disc_B_losses):.3f}, \
                Train Gen Loss: {sum(gen_AB_losses) / len(gen_AB_losses):.3f}'
        )
