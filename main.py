from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from DCVAE import DCVAE
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='DCVAE')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--image-size', type=int, default=128, metavar='N',
                    help='image-size (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--runs', type=int, default=1, metavar='S',
                    help='time of runs(default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
writer = SummaryWriter()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Data loading code

ngf = 64
nz = 300
nc = 3

traindir = os.path.join(args.data, 'nevus')

normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ]))


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


model = DCVAE(args.image_size, ngf, nz, nc).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = nn.BCELoss()(recon_x, x)
    BCE = nn.MSELoss(reduction='sum')(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + 0.01 * KLD
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE_Loss: {:.6f}\tKL_Loss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                BCE.item() / len(data), KLD.item() / len(data),
                loss.item() / len(data)))
        if batch_idx == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 3, args.image_size, args.image_size)[:n]])
            save_image(comparison.cpu(),
                       os.path.abspath('results/reconstruction_') + str(epoch) + '_runs' + str(args.runs) + '.png',
                       nrow=n)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    writer.add_scalar('Training Loss', train_loss / len(train_loader.dataset), epoch)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'train_loss': train_loss / len(train_loader.dataset),
        'optimizer': optimizer.state_dict(),
    }
    filename = 'checkpoint_runs' + str(args.runs) + '.pth.tar'
    torch.save(state, filename)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(16, nz, 1, 1).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(16, 3, args.image_size, args.image_size),
                       os.path.abspath('results/sample_') + '_runs' + str(args.runs) + '_epoch'+str(epoch) + '.png')

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()