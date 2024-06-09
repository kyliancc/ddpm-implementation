import torch
from torchvision.utils import save_image

import argparse
from datetime import datetime
import os

from unet import DDPMUNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-f', type=str, default=None, help='path to checkpoint', required=True)
    parser.add_argument('--output', '-o', type=str, default='./out/', help='output directory')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--number', '-n', type=int, default=4, help='how many images to generate')
    parser.add_argument('--width', '-x', type=int, default=96, help='width of generated image')
    parser.add_argument('--height', '-y', type=int, default=96, help='height of generated image')
    parser.add_argument('--max_time_step', '-t', type=int, default=1000, help='how many time steps of Markov chain')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args.__dict__}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    model = DDPMUNet(in_channels=3, out_channels=3, t_channels=128)
    model.to(device)
    state_dict = torch.load(args.load)['model']
    model.load_state_dict(state_dict)

    model.eval()

    betas = torch.linspace(1e-4, 0.02, args.max_time_step, device=device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    num = 0
    print('START INFERENCE')
    for i in range(args.number // args.batch_size):
        batch_size = args.batch_size
        if i == args.number // args.batch_size:
            batch_size = args.number % args.batch_size

        x = torch.randn([batch_size, 3, args.height, args.width], device=device)
        for t in range(args.max_time_step, 0, -1):
            with torch.no_grad():
                beta = betas[t-1]
                alpha = alphas[t-1]
                alpha_bar = alphas_bar[t-1]

                pred = model(x, torch.full([batch_size,], t, device=device))
                x_next = (1 / torch.sqrt(alpha)) * (x - pred * ((1 - alpha) / torch.sqrt(1 - alpha_bar)))

                if t > 1:
                    z = torch.randn_like(x, device=device)
                    alpha_bar_pre = alphas_bar[t-2]
                    x_next += z * torch.sqrt((1 - alpha_bar_pre) / (1 - alpha_bar) * beta)

                x = x_next
                print(f'Denoised from time step {t}.')

        for j in range(batch_size):
            img = x[j,:,:,:]
            save_path = os.path.join(args.output, f'{current_time}-{num}.png')
            save_image(img, save_path)
            print(f'Image saved to {save_path}')
            num += 1


if __name__ == '__main__':
    main()
