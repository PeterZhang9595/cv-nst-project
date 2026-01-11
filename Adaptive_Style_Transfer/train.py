import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

import net
from net import VGGEncoder, Decoder, AdaIN, StyleTransferNet


# -------------------------
# Transform
# -------------------------
def train_transform():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])


# -------------------------
# Dataset
# -------------------------
class DatasetFromFolder(data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.image_filenames = [
            p for p in self.root.glob('*')
            if p.suffix.lower() in ['.jpg', '.png', '.jpeg']
        ]

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index]).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.image_filenames)


def main():

    # -------------------------
    # Args
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, required=True)
    parser.add_argument('--style_dir', type=str, required=True)

    parser.add_argument('--save_dir', default='./save')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--save_model_interval', type=int, default=10000)

    args = parser.parse_args()


    # -------------------------
    # Setup
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    # -------------------------
    # Network
    # -------------------------
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features # only use the conv part 
    encoder = VGGEncoder(vgg) # fixed encoder
    decoder = Decoder() # to be trained 

    network = StyleTransferNet(encoder,decoder).to(device)
    network.train()

    # dataloader 
    content_loader = data.DataLoader(
        DatasetFromFolder(args.content_dir, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        drop_last=True
    )

    style_loader = data.DataLoader(
        DatasetFromFolder(args.style_dir, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        drop_last=True
    )

    content_iter = iter(content_loader)
    style_iter = iter(style_loader)

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter)


    # -------------------------
    # Training loop
    # -------------------------
    for i in tqdm(range(args.max_iter)):
        try:
            content_images = next(content_iter)
        except StopIteration:
            content_iter = iter(content_loader)
            content_images = next(content_iter)

        try:
            style_images = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_images = next(style_iter)

        content_images = content_images.to(device)
        style_images = style_images.to(device)

        loss_c, loss_s = network(content_images, style_images)

     
        loss = (
            args.content_weight * loss_c +
            args.style_weight * loss_s
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('loss/content', loss_c.item(), i + 1)
        writer.add_scalar('loss/style', loss_s.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0:
            torch.save(
                network.decoder.state_dict(),
                f'{save_dir}/decoder_iter_{i+1}.pth'
            )
            print(f'iteration {i+1}: loss_c={loss_c.item():.2f}, loss_s={loss_s.item():.2f}')
    # store the final model
    torch.save(
        network.decoder.state_dict(),
        f'{save_dir}/decoder.pth'
    )

    writer.close()

if __name__ == '__main__':
    main()