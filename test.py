import argparse
import torch
import os
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import nn
from DCVAE import DCVAE
from sklearn import metrics
parser = argparse.ArgumentParser(description='DCVAE')
parser.add_argument('--data', metavar='DIR', type=str, default='all_classes',
                    help='path to dataset')
parser.add_argument('--resume', default='checkpoint_runs3.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--image-size', type=int, default=128, metavar='N',
                    help='image-size (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
model = DCVAE(128, 64, 300, 3).to(device)
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'])

testdir = os.path.join(args.data, 'test')

normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))


test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    pin_memory=True)

class_to_idx = datasets.ImageFolder(testdir).class_to_idx

model.eval()


# Reconstruction + KL divergence losses summed over all elements and batch
def pixel_loss_function(recon_x, x, mu, logvar):
    # BCE = nn.BCELoss()(recon_x, x)
    BCE = nn.MSELoss(reduction='none')(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


def get_auc(index1, index2):
    y1_true = np.ones(sum(index1))
    y2_true = np.zeros(sum(index2))
    y = np.append(y1_true, y2_true)
    y1_target = result[2, index1]
    y2_target = result[2, index2]
    pred = np.append(y1_target, y2_target)
    return metrics.roc_auc_score(y, pred)



with torch.no_grad():
     result = np.empty(shape=[3, 0])
     index = np.empty(shape=[0], dtype=int)
     for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        pix_BCE, pix_KLD = pixel_loss_function(recon_batch, data, mu, logvar)
        BCE = torch.mean(pix_BCE, (1, 2, 3))
        KLD = torch.mean(pix_KLD, 1)
        s_BCE = torch.exp(-BCE)
        s_KLD = torch.exp(-KLD)
        loss = BCE + KLD
        s_total = torch.exp(-loss)
        x = torch.stack((s_BCE, s_KLD, s_total)).cpu().detach().numpy()
        result = np.hstack([result, x])
        index = np.append(index, target.numpy())
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                     recon_batch.view(args.batch_size, 3, args.image_size, args.image_size)[:n]])
        save_image(comparison.cpu(),
                    os.path.abspath('results/reconstruction_') + str(i) + '.png', nrow=n)

rocauc = []
index2 = (index==20)
for i in range(26):
    if i!=20:
        index1 = (index==i)
        rocauc.append(get_auc(index1, index2))

with open('rocauc.txt', 'w') as f:
    for item in rocauc:
        f.write("%s\n" % item)
