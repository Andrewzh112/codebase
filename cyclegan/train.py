import torch
import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import itertools
from tqdm import tqdm

from networks.utils import load_weights
from cyclegan.data import CycleImageDataset
from cyclegan.models import Generator, Discriminator, set_requires_grad
from cyclegan.utils import ReplayBuffer, LambdaLR, make_images, get_random_ids


parser = argparse.ArgumentParser()
parser.add_argument('--dim_A', type=int, default=3, help='Numer of channels for class A')
parser.add_argument('--dim_B', type=int, default=3, help='Numer of channels for class B')
parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of ResNet Blocks for generators')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for model')
parser.add_argument('--lr_G', type=float, default=0.0002, help='Learning rate for generators')
parser.add_argument('--lr_D', type=float, default=0.0002, help='Learning rate for discriminators')
parser.add_argument('--continue_train', action="store_true", default=False, help='continue training')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--starting_epoch', type=int, default=0, help='Starting epoch for resuming training')
parser.add_argument('--decay_epoch', type=int, default=50, help='Starting epoch when learning rate will start decay')
parser.add_argument('--load_shape', type=int, default=256, help='Initial image H or W')
parser.add_argument('--target_shape', type=int, default=224, help='Final image H or W')
parser.add_argument('--progress_interval', type=int, default=1, help='Save model and generated image every x epoch')
parser.add_argument('--sample_batches', type=int, default=32, help='How many generated images to sample')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--lambda_identity', type=float, default=0.1, help='Identity loss weight')
parser.add_argument('--lambda_cycle', type=float, default=10., help='Cycle loss weight')
parser.add_argument('--log_dir', type=str, default='cyclegan/logs', help='Path to where log files will be saved')
parser.add_argument('--save_img_dir', type=str, default='cyclegan/save_images', help='Path to where generated images will be saved')
parser.add_argument('--data_root', type=str, default='data/horse2zebra', help='Path to where image data is located')
parser.add_argument('--checkpoint_dir', type=str, default='cyclegan/model_weights', help='Path to where model weights will be saved')
args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(
            args.log_dir + f"/{args.data_root.split('/')[-1]}{int(datetime.now().timestamp()*1e6)}")
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
    optimizer_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(),
        D_B.parameters()),
        lr=args.lr_D,
        betas=args.betas
    )

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
    )
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D,
        lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
    )

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    pool_A = ReplayBuffer()
    pool_B = ReplayBuffer()

    if args.continue_train:
        args.start_epoch = load_weights(state_dict_path=f"{args.checkpoint_dir}/{args.data_root.split('/')[-1]}.pth",
                                        models=[D_A, D_B, G_AB, G_BA],
                                        model_names=['D_A', 'D_B', 'G_AB', 'G_BA'],
                                        optimizers=[optimizer_G, optimizer_D],
                                        optimizer_names=['optimizer_G', 'optimizer_D'],
                                        return_val='start_epoch')
    pbar = tqdm(
            range(args.starting_epoch, args.n_epochs),
            total=(args.n_epochs - args.starting_epoch)
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    sampled_idx = get_random_ids(len(dataloader), args.sample_batches)
    for epoch in pbar:
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        disc_A_losses, gen_A_losses, disc_B_losses, gen_B_losses = [], [], [], []
        (gen_ad_A_losses,
         gen_ad_B_losses,
         gen_id_A_losses,
         gen_id_B_losses,
         gen_cyc_A_losses,
         gen_cyc_B_losses) = [], [], [], [], [], [] 
        real_As, real_Bs, fake_As, fake_Bs = [], [], [], []
        identity_As, identity_Bs, cycle_As, cycle_Bs = [], [], [], []
        for batch_idx, (real_A, real_B) in enumerate(dataloader):
            real_A = torch.nn.functional.interpolate(real_A, size=args.target_shape).to(device)
            real_B = torch.nn.functional.interpolate(real_B, size=args.target_shape).to(device)

            ## generator forward pass ##
            fake_A = G_BA(real_B)
            fake_B = G_AB(real_A)
            cycle_A = G_BA(fake_B)
            cycle_B = G_AB(fake_A)
            identity_A = G_BA(real_A)
            identity_B = G_AB(real_B)

            # disc forward pass
            fake_A_logits = D_A(fake_A)
            fake_B_logits = D_B(fake_B)
            real_A_logits = D_A(real_A)
            real_B_logits = D_B(real_B)

            # sample from queue
            pool_fake_A = pool_A.sample(fake_A.clone().detach())
            pool_fake_B = pool_B.sample(fake_B.clone().detach())
            fake_pool_A_logits = D_A(pool_fake_A)
            fake_pool_B_logits = D_B(pool_fake_B)

            # disc loss
            disc_A_fake_loss = criterion_GAN(fake_pool_A_logits, torch.zeros_like(fake_pool_A_logits))
            disc_A_real_loss = criterion_GAN(real_A_logits, torch.ones_like(real_A_logits))
            disc_A_loss = (disc_A_fake_loss + disc_A_real_loss) / 2
            disc_B_fake_loss = criterion_GAN(fake_pool_B_logits, torch.zeros_like(fake_pool_B_logits))
            disc_B_real_loss = criterion_GAN(real_B_logits, torch.ones_like(real_B_logits))
            disc_B_loss = (disc_B_fake_loss + disc_B_real_loss) / 2
            disc_loss = disc_A_loss + disc_B_loss

            # generator loss
            adversarial_A_loss = criterion_GAN(fake_A_logits, torch.ones_like(fake_A_logits))
            adversarial_B_loss = criterion_GAN(fake_B_logits, torch.ones_like(fake_B_logits))
            cycle_A_loss = criterion_cycle(cycle_A, real_A)
            cycle_B_loss = criterion_cycle(cycle_B, real_B)
            identity_A_loss = criterion_identity(identity_A, real_A)
            identity_B_loss = criterion_identity(identity_B, real_B)
            gen_A_loss = adversarial_A_loss + args.lambda_identity*identity_A_loss + args.lambda_cycle*cycle_A_loss
            gen_B_loss = adversarial_B_loss + args.lambda_identity*identity_B_loss + args.lambda_cycle*cycle_B_loss
            gen_loss = gen_A_loss + gen_B_loss

            # update gens
            set_requires_grad([D_A, D_B], False)
            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()

            # update discs
            set_requires_grad([D_A, D_B], True)
            optimizer_D.zero_grad()
            disc_loss.backward()
            optimizer_D.step()

            # log
            gen_A_losses.append(gen_A_loss.item())
            gen_B_losses.append(gen_B_loss.item())
            disc_A_losses.append(disc_A_loss.item())
            disc_B_losses.append(disc_B_loss.item())
            gen_ad_A_losses.append(adversarial_A_loss.item())
            gen_ad_B_losses.append(adversarial_B_loss.item())
            gen_id_A_losses.append(identity_A_loss.item())
            gen_id_B_losses.append(identity_B_loss.item())
            gen_cyc_A_losses.append(cycle_A_loss.item())
            gen_cyc_B_losses.append(cycle_B_loss.item())
            if batch_idx in sampled_idx:
                real_As.append(real_A.detach().cpu())
                real_Bs.append(real_B.detach().cpu())
                fake_As.append(fake_A.detach().cpu())
                fake_Bs.append(fake_B.detach().cpu())
                identity_As.append(identity_A.detach().cpu())
                identity_Bs.append(identity_B.detach().cpu())
                cycle_As.append(cycle_A.detach().cpu())
                cycle_Bs.append(cycle_B.detach().cpu())

        lr_scheduler_D.step()
        lr_scheduler_G.step()

        if (epoch + 1) % args.progress_interval == 0:
            writer.add_scalars('Train Losses', {
                    'Discriminator A': sum(disc_A_losses) / len(disc_A_losses),
                    'Discriminator B': sum(disc_B_losses) / len(disc_B_losses),
                    'Generator A': sum(gen_A_losses) / len(gen_A_losses),
                    'Generator B': sum(gen_B_losses) / len(gen_B_losses),
                    'Generator Adversarial A': sum(gen_ad_A_losses) / len(gen_ad_A_losses),
                    'Generator Adversarial B': sum(gen_ad_B_losses) / len(gen_ad_B_losses),
                    'Generator Cycle A': sum(gen_cyc_A_losses) / len(gen_cyc_A_losses),
                    'Generator Cycle B': sum(gen_cyc_B_losses) / len(gen_cyc_B_losses),
                    'Generator Identity A': sum(gen_id_A_losses) / len(gen_id_A_losses),
                    'Generator Identity B': sum(gen_id_B_losses) / len(gen_id_B_losses)
                }, global_step=epoch)
            writer.add_image('Fake A', make_images(fake_As), global_step=epoch)
            writer.add_image('Fake B', make_images(fake_Bs), global_step=epoch)
            writer.add_image('Real A', make_images(real_As), global_step=epoch)
            writer.add_image('Real B', make_images(real_Bs), global_step=epoch)
            writer.add_image('Identity A', make_images(identity_As), global_step=epoch)
            writer.add_image('Identity B', make_images(identity_Bs), global_step=epoch)
            writer.add_image('Cycle A', make_images(cycle_As), global_step=epoch)
            writer.add_image('Cycle B', make_images(cycle_Bs), global_step=epoch)
            if (epoch + 1) % 10 == 0:
                if not os.path.isdir(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                torch.save({
                    'G_AB': G_AB.state_dict(),
                    'G_BA': G_BA.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'D_A': D_A.state_dict(),
                    'D_B': D_B.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'start_epoch': epoch + 1
                }, f"{args.checkpoint_dir}/{args.data_root.split('/')[-1]}.pth")
        tqdm.write('#########################################################')
        tqdm.write(
            f'Epoch {epoch + 1}/{args.n_epochs}, \
                Discriminator A: {sum(disc_A_losses) / len(disc_A_losses):.3f}, \
                Discriminator B: {sum(disc_B_losses) / len(disc_B_losses):.3f}, \
                Generator A: {sum(gen_A_losses) / len(gen_A_losses):.3f}, \
                Generator B: {sum(gen_B_losses) / len(gen_B_losses):.3f}'
        )
