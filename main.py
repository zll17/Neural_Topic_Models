import argparse

import matplotlib.pyplot as plt
from munkres import Munkres
from sklearn.manifold import TSNE
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from vade import VaDE, lossfun

N_CLASSES = 10
PLOT_NUM_PER_CLASS = 128


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()

    total_loss = 0
    for x, _ in data_loader:
        x = x.to(device).view(-1, 784)
        recon_x, mu, logvar = model(x)
        loss = lossfun(model, x, recon_x, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch:{},Loss:{}'.format(epoch,total_loss/len(data_loader)))
    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)


def test(model, data_loader, device, epoch, writer, plot_points):
    model.eval()

    gain = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int, device=device)
    with torch.no_grad():
        for xs, ts in data_loader:
            xs, ts = xs.to(device).view(-1, 784), ts.to(device)
            ys = model.classify(xs)
            for t, y in zip(ts, ys):
                gain[t, y] += 1
        cost = (torch.max(gain) - gain).cpu().numpy()
        assign = Munkres().compute(cost)
        acc = torch.sum(gain[tuple(zip(*assign))]).float() / torch.sum(gain)

        # Plot latent space
        xs, ts = plot_points[0].to(device), plot_points[1].numpy()
        zs = model.encode(xs)[0].cpu().numpy()
        tsne = TSNE(n_components=2, init='pca')
        zs_tsne = tsne.fit_transform(zs)

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for t in range(10):
            points = zs_tsne[ts == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t))
        ax.legend()

    writer.add_scalar('Acc/test', acc.item(), epoch)
    writer.add_figure('LatentSpace', fig, epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=100)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.002)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=512)
    parser.add_argument('--pretrain', '-p',
                        help='Load parameters from pretrained model.',
                        type=str, default=None)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=if_use_cuda)

    # For plotting
    plot_points = {}
    for t in range(10):
        points = torch.cat([data for data, label in dataset if label == t])
        points = points.view(-1, 784)[:PLOT_NUM_PER_CLASS].to(device)
        plot_points[t] = points
    xs = []
    ts = []
    for t, x in plot_points.items():
        xs.append(x)
        t = torch.full((x.size(0),), t, dtype=torch.long)
        ts.append(t)
    plot_points = (torch.cat(xs, dim=0), torch.cat(ts, dim=0))

    model = VaDE(N_CLASSES, 784, 10)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # LR decreases every 10 epochs with a decay rate of 0.9
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, data_loader, optimizer, device, epoch, writer)
        test(model, data_loader, device, epoch, writer, plot_points)
        lr_scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
