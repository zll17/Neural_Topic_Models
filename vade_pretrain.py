import argparse

from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms
from data import *
from vade import AutoEncoderForPretrain, VaDE

def train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for i,x in enumerate(data_loader):
        batch_size = x.size(0)
        x = x.to(device)
        recon_x = model(x)
        loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {:>3}: Train Loss = {}'.format(
        epoch, total_loss / len(data_loader)))


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=20)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.01)
    parser.add_argument('--batch_size', '-b',
                        help='Batch size.',
                        type=int, default=256)
    parser.add_argument('--out', '-o',
                        help='Output path.',
                        type=str, default='./vade_parameter.pth')
    parser.add_argument('--n_topic', 
                        help='num of topics',
                        type=int, default=30)
    parser.add_argument('--taskname', 
                        help='taskname',
                        type=str, default='sub')
    parser.add_argument('--use_stopwords', 
                        help='whether to apply stopwords(1/0)',
                        type=bool, default=True)


    args = parser.parse_args()
    
    batch_size = args.batch_size
    n_topic = args.n_topic
    taskname = args.taskname
    num_epochs = args.num_epochs
    use_stopwords = args.use_stopwords

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    print('Loading train_data ...')
    train_loader,vocab,txtDocs = get_batch(taskname,use_stopwords,batch_size)

    pretrain_model = AutoEncoderForPretrain(data_dim=len(vocab), latent_dim=n_topic).to(device)

    optimizer = torch.optim.Adadelta(pretrain_model.parameters(),
                                 lr=args.learning_rate)


    #pretrain_model = nn.DataParallel(pretrain_model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)
    for epoch in range(1, num_epochs + 1):
        train(pretrain_model, train_loader, optimizer, device, epoch)
        lr_scheduler.step()
    #pretrain_model = pretrain_model.module

    with torch.no_grad():
        z = []
        for data in train_loader:
            data = data.to(device)
            it = pretrain_model.encode(data).cpu()
            z.append(it)
        z = torch.cat(z,dim=0)


    pretrain_model = pretrain_model.cpu()
    state_dict = pretrain_model.state_dict()

    print('GMM fitting ...')
    gmm = GaussianMixture(n_components=n_topic, covariance_type='diag',max_iter=1000)
    gmm.fit(z)

    model = VaDE(n_topic, len(vocab), n_topic)
    model.load_state_dict(state_dict, strict=False)
    model._pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()
    model.mu.data = torch.from_numpy(gmm.means_).float()
    #model.logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()
    tmp = torch.log(torch.from_numpy(gmm.covariances_)).float()
    epsilon,_ = torch.min(tmp,dim=1,keepdim=True)
    model.logvar.data = tmp - epsilon
    print('gmm.covariances:',gmm.covariances_)
    print('model.logvar:',model.logvar)

    out_params_name = 'premodel/VaDE_{}_K{}_params.pth'.format(taskname,n_topic)
    print('Saving model...')
    torch.save(model.state_dict(), out_params_name)


if __name__ == '__main__':
    main()
