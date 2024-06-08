import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse

from dataset import AnimeFaceDataset
from unet import DDPMUNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--lr', '-l', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=1, help='epochs to train')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='how many dataloader workers')
    parser.add_argument('--load', '-f', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--save', '-s', type=int, default=200, help='how many iterations to save')
    parser.add_argument('--max_time_step', '-t', type=int, default=1000, help='how many time steps of Markov chain')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args.__dict__}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_dataset = AnimeFaceDataset('train', resize=96)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = DDPMUNet(in_channels=3, out_channels=3, t_channels=128, device=device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    iterations = 0

    if args.load:
        state_dict = torch.load(args.load)
        iterations = state_dict['iterations']
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded checkpoint from {args.load}.')

    print('START TRAINING')

    betas = torch.linspace(1e-4, 0.02, args.max_time_step, device=device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    for epoch in range(args.epochs):
        for img in train_loader:
            model.train()
            optimizer.zero_grad()

            img = img.to(device)

            time_step = torch.randint(1, args.max_time_step + 1, [img.size(0),], device=device)
            noise = torch.randn_like(img, device=device)

            mean = img * torch.reshape(torch.sqrt(alphas_bar[time_step-1]), [-1, 1, 1, 1])
            std = torch.reshape(torch.sqrt(1 - alphas_bar[time_step-1]), [-1, 1, 1, 1])

            xt = mean + std * noise
            y = model(xt, time_step)
            loss = criterion(y, noise)
            loss.backward()

            optimizer.step()

            print(f'Iteration {iterations} finished with loss: {loss.item()}')
            iterations += 1

            if iterations % args.save == 0:
                state_dict = {
                    'iterations': iterations,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state_dict, f'./checkpoints/ddpm-unet-{iterations}.pth')
                print(f'Checkpoint saved to ./checkpoints/ddpm-unet-{iterations}.pth')


if __name__ == '__main__':
    main()
