import argparse
import deepspeed
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from zero_offload.vit_pytorch import ViT
from time import perf_counter


def add_argument():
    """
    https://www.deepspeed.ai/tutorials/cifar-10/
    """
    parser=argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def main():
    args = add_argument()
    dataset = CIFAR10('.', download=True, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=8)
    huge_model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    lr = 0.001
    warmup_steps = 1000
    remain_steps = (args.epochs * len(trainloader) - warmup_steps)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        huge_model.parameters(),
        lr=lr,
        betas=(0.8, 0.999),
        eps=1e-8,
        weight_decay=3e-7)
    torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (epoch + 1) / warmup_steps * lr if epoch < warmup_steps else (epoch - warmup_steps) * lr / remain_steps)
    model_engine, _, trainloader_ds, _ = deepspeed.initialize(
        args=args,
        model=huge_model,
        model_parameters=huge_model.parameters(),
        training_data=dataset)

    # training w/ DeepSpeed
    start_time = perf_counter()
    for data in trainloader_ds:
         inputs = data[0].to(model_engine.device)
         labels = data[1].to(model_engine.device)

         outputs = model_engine(inputs)
         loss = criterion(outputs, labels)

         model_engine.backward(loss)
         model_engine.step()
    ds_time = (perf_counter() - start_time) / 60
    print('###################################################################')
    print(f'Training CIFAR10 using DeepSpeed used {ds_time:.3f} minutes')

    # regular training
    model = huge_model.to(device)
    start_time = perf_counter()
    for data in trainloader:
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    no_ds_time = (perf_counter() - start_time) / 60
    print('###################################################################')
    print(f'Training CIFAR10 without using DeepSpeed used {no_ds_time:.3f} minutes')
    print('###################################################################')
    print(f'DeepSpeed accelerated training by {no_ds_time - ds_time:.3f} minutes')


if __name__ == '__main__':
    main()
