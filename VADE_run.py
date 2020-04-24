import os
import re
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gensim
from sklearn.manifold import TSNE
from munkres import Munkres
from data import *
from utils import *
from vade import VaDE  #,lossfun

parser = argparse.ArgumentParser(description='Neural Topic Model')

# 添加参数步骤
parser.add_argument('--taskname', type=str,default='sub',help='Name of the task e.g subX')
parser.add_argument('--n_topic', type=int,default=20,help='Number of the topics')
parser.add_argument('--num_epochs',type=int,default=2,help='Num of epochs')
parser.add_argument('--batch_size',type=int,default=1,help='Batch Size')
parser.add_argument('--gpu',type=str,default='0',help='GPU device e.g 1')
parser.add_argument('--ratio',type=float,default=1.0,help='Ratio of the train data for actual use')
parser.add_argument('--use_stopwords',type=bool,default=True,help='Whether to use stopwords or not')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load the trained model and continue to train')
parser.add_argument('--pretrain',type=str,default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

n_topic = args.n_topic
taskname = args.taskname
num_epochs = args.num_epochs
batch_size = args.batch_size
ratio = args.ratio
use_stopwords = args.use_stopwords
bkpt_continue = args.bkpt_continue
pretrain = args.pretrain if args.pretrain!=None else 'premodel/VaDE_{}_K{}_params.pth'.format(taskname,n_topic)
print('bkpt:',bkpt_continue)

model_name = 'VaDE'
msg = 'BCE'
run_name = '{}_K{}_{}_{}'.format(model_name,n_topic,taskname,msg)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def _reparameterize(mu, logvar):
    """Reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


def lossfun(model, x, recon_x, mu, logvar):
    batch_size = x.size(0)
            
    # Compute gamma ( q(c|x) )
    z = _reparameterize(mu, logvar).unsqueeze(1)
    #print('z:',z)
    h = z - model.mu
    #print('h1:',h)
    h = torch.exp(-0.5 * torch.sum((h * h / model.logvar.exp()), dim=2))
    #print('h2:',h)
    # Same as `torch.sqrt(torch.prod(model.logvar.exp(), dim=1))`
    #print('model.logvar:',model.logvar)
    #print('torch.sum(0.5 * model.logvar, dim=1).exp():',torch.sum(0.5 * model.logvar, dim=1).exp())
    #print('torch.sqrt(torch.prod(model.logvar.exp(), dim=1)):',torch.sqrt(torch.prod(model.logvar.exp(), dim=1)))
    h = h / torch.sum(0.5 * model.logvar, dim=1).exp()
    #print('h3:',h)
    p_z_given_c = h / (2 * math.pi)
    #print('p(z|c):',p_z_given_c)
    p_z_c = p_z_given_c * model.weights
    #print('p_z_c:',p_z_c)
    gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
    #print('gamma:',gamma)

    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
    #print('h4:',h)
    h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
    #print('h5:',h)
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #print('loss1:',loss)
    loss2 = 0.5 * torch.sum(gamma * h) \
        - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
        + torch.sum(gamma * torch.log(gamma + 1e-9)) \
        - 0.5 * torch.sum(1 + logvar)
    #print('loss2:',loss2)
    loss += loss2
    loss = loss / batch_size
    return loss

print('Loading train_data ...')
train_loader,vocab,txtDocs = get_batch(taskname,use_stopwords,batch_size)

# Hyper-parameters
learning_rate = 1e-3

model = VaDE(n_classes=n_topic,data_dim=len(vocab),latent_dim=n_topic).to(device)
model = model.to(device)
if pretrain:
    model.load_state_dict(torch.load(pretrain))
#model = nn.DataParallel(model)
if bkpt_continue:
    print('Loading parameters of the model...')
    model.load_state_dict(torch.load('ckpt/{}.model'.format(run_name)))
#model = model.module
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)

if bkpt_continue:
    logger = open('logs/{}.log'.format(run_name),'a',encoding='utf-8')
    logger.write('\n')
else:
    logger = open('logs/{}.log'.format(run_name),'w',encoding='utf-8')
    logger.write(str(model)+'\n')


def show_topics(model=model, n_topic=n_topic, topK=20, showWght=False,fix_topic=None):
    global vocab
    if isinstance(model,nn.DataParallel):
        model = model.module
    topics = []
    if fix_topic!=None:
        idxes = model.mu[fix_topic].unsqueeze(0).to(device)
    else:
        idxes = model.mu.to(device)
    word_dists = model.decode(idxes)
    vals,indices = torch.topk(word_dists,topK,dim=1)
    vals = vals.cpu().tolist()
    indices = indices.cpu().tolist()
    if fix_topic is None:
        for i in range(n_topic):
            if showWght==True:
                topics.append([(wght,vocab.id2token[idx]) for wght,idx in zip(vals[i],indices[i])])
            else:
                topics.append([vocab.id2token[idx] for idx in indices[i]])
    else:
        if showWght==True:
            topics.append([(wght,vocab.id2token[idx]) for wght,idx in zip(vals[fix_topic],indices[fix_topic])])
        else:
            topics.append([vocab.id2token[idx] for idx in indices[fix_topic]])
    return topics


print('Start training...')
def train(model, data_loader, num_epochs,writer):
    model.train()
    logger.write('='*30+'Loss'+'='*30+'\n')
    for epoch in range(num_epochs):
        total_loss = 0
        for i, x in enumerate(data_loader):
            # Customized Train Process
            # Forward pass
            x = x.to(device)
            x_reconst, mu, log_var = model(x)
            '''
            print('step ',i)
            print('before bp,model._pi:',model._pi)
            print('model.mu:',model.mu)
            print('before bp,x[0]:',x[0])
            print('before bp,x_reconst[0]:',x_reconst[0])
            '''
            #loss = F.binary_cross_entropy(x_reconst,x,reduction='sum')
            loss = lossfun(model,x,x_reconst,mu,log_var)
            #print('loss:',loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('after bp,model._pi:',model._pi)

            if (i+1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Loss: {:.8f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()/batch_size))

                logger.write("Epoch[{}/{}], Step [{}/{}], Loss: {:.8f}\n".format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()/batch_size))
                torch.save(model.state_dict(), 'ckpt/{}.model'.format(run_name))
        print('==Epoch{},Loss:{}'.format(epoch,total_loss/len(data_loader)))
        writer.add_scalar('Loss/train',total_loss/len(data_loader),epoch)

        if (epoch+1)%10==0:
            topic_text = show_topics(model)
            print('Epoch {}:'.format(epoch))
            for tp in topic_text:
                print(tp)
        #lr_scheduler.step()
    logger.write('='*60+'\n\n')
    return 

writer = SummaryWriter()
train(model ,train_loader, num_epochs=num_epochs,writer=writer)
writer.close()

model.eval()

if isinstance(model,nn.DataParallel):
    model = model.module
torch.save(model.state_dict(), 'ckpt/{}.model'.format(run_name))
print('model saved')

import jieba

def infer_topic(strText,model):
    bow = doc2bow(list(jieba.cut(strText)))
    bow = bow.unsqueeze(0)
    return model.encode(bow.cuda())[0].argmax(1)

def infer_topic_tensor(bow,model):
    return model.encode(bow.cuda())[0].argmax(1)

#infer_topic('你或许需要做个核磁共振',model)
#infer_topic_tensor(next(iter(train_loader)),model)

topics_text=show_topics()

# Evaluate the Model with Topic Coherence
from gensim.models import CoherenceModel

# Computing the C_V score
cv_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_v')
cv_coherence_score = cv_coherence_model.get_coherence_per_topic()
cv_coherence_avg = cv_coherence_model.get_coherence()

# Computing Topic Diversity score
topic_diversity = calc_topic_diversity(topics_text)

# Computing the C_W2V score

try:
    if os.path.exists('data/{}/{}_embeddings.txt'.format(taskname,taskname)):
        keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format('data/{}/{}_embeddings.txt'.format(taskname,taskname),binary=False)
        w2v_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_w2v',keyed_vectors=keyed_vectors)
    else:
        w2v_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_w2v')
    w2v_coherence_score = w2v_coherence_model.get_coherence_per_topic()
    w2v_coherence_avg = w2v_coherence_model.get_coherence()
except:
    #In case of OOV Error
    w2v_coherence_score = cv_coherence_score
    w2v_coherence_avg = cv_coherence_avg


# Computing the C_UCI score
c_uci_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_uci')
c_uci_coherence_score = c_uci_coherence_model.get_coherence_per_topic()
c_uci_coherence_avg = c_uci_coherence_model.get_coherence()

# Computing the C_NPMI score
c_npmi_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_npmi')
c_npmi_coherence_score = c_npmi_coherence_model.get_coherence_per_topic()
c_npmi_coherence_avg = c_npmi_coherence_model.get_coherence()


logger.write('Topics:\n')

for tp,cv,w2v in zip(topics_text,cv_coherence_score,w2v_coherence_score):
    print('{}+++$+++cv:{}+++$+++w2v:{}'.format(tp,cv,w2v))
    logger.write('{}+++$+++cv:{}+++$+++w2v:{}\n'.format(str(tp),cv,w2v))

print('c_v for ${}$: {}'.format(run_name,cv_coherence_avg))
logger.write('c_v for ${}$: {}\n'.format(run_name, cv_coherence_avg))

print('c_w2v for ${}$: {}'.format(run_name,w2v_coherence_avg))
logger.write('c_w2v for ${}$: {}\n'.format(run_name, w2v_coherence_avg))

print('c_uci for ${}$: {}'.format(run_name,c_uci_coherence_avg))
logger.write('c_uci for ${}$: {}\n'.format(run_name, c_uci_coherence_avg))

print('c_npmi for ${}$: {}'.format(run_name,c_npmi_coherence_avg))
logger.write('c_npmi for ${}$: {}\n'.format(run_name, c_npmi_coherence_avg))

print('t_div for ${}$: {}'.format(run_name,topic_diversity))
logger.write('t_div for ${}$: {}\n'.format(run_name, topic_diversity))


logger.close()
exit(0)


print('Run f-VAE-g')
# CVAE model
class VAE_Gumbel(nn.Module):
    def __init__(self, bow_size=len(dictionary), h_dim=1024, z_dim=n_topic, n_classes=n_topic):
        super(VAE_Gumbel, self).__init__()
        self.fc1 = nn.Linear(bow_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(h_dim, n_classes)
        self.fc5 = nn.Linear(z_dim + n_classes, h_dim)
        self.fc6 = nn.Linear(h_dim, bow_size)  
        self.fc7 = nn.Linear(z_dim,n_classes)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, log_var, log_p = self.fc2(h), self.fc3(h), F.log_softmax(self.fc4(h), dim=1)
        return mu, log_var, log_p
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        theta = torch.softmax(self.fc7(z),dim=1)
        #theta = torch.softmax(z,dim=1)
        return theta

    def decode(self, z, y_onehot):
        latent = torch.cat((z, y_onehot),dim=1)
        h = F.relu(self.fc5(latent))
        x_reconst = torch.sigmoid(self.fc6(h))
        return x_reconst
    
    def forward(self, x):
        mu, log_var, log_p = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_onehot = F.gumbel_softmax(log_p)
        x_reconst = self.decode(z, y_onehot)
        #print(alpha)
        return x_reconst, mu, log_var, log_p


# Hyper-parameters
learning_rate = 1e-3
n_classes = n_topic

model_G = VAE_Gumbel().to(device)
optimizer = torch.optim.Adam(model_G.parameters(), lr=learning_rate)


def to_onehot(labels,n_class=n_topic):
    labels = labels.unsqueeze(1)
    return torch.zeros(labels.shape[0], n_class).to(device).scatter_(1, labels, 1)

#to_onehot(torch.tensor([19, 18,  6, 18]).to(device))


from sklearn.metrics import normalized_mutual_info_score

def train_G_modified_loss(model, preModel, data_loader, num_epochs, beta=1 , C_z_fin=0, C_c_fin=0, verbose=True):
    nmi_scores = []
    model.train(True)
    NMI_history = []
    rec_loss = []
    kl_loss = []
    L = len(data_loader)
    logger = open('logs/{}_{}_f-vae-g.log'.format(taskname,n_topic),'a',encoding='utf-8')
    for epoch in range(num_epochs):
        
        C_z = C_z_fin * epoch/num_epochs
        C_c = C_c_fin * epoch/num_epochs
        
        for i, x in enumerate(data_loader):
            # Forward pass
            x = x.to(device)
            labels = infer_topic_tensor(x,preModel)
            x_reconst, mu, log_var, log_p = model(x)
            p = torch.exp(log_p)
            
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            p_uniform = 1/n_classes * torch.ones_like(p)
            entropy = torch.sum(p * torch.log(p / p_uniform))
            NMI = normalized_mutual_info_score(labels.cpu().numpy(), p.cpu().max(1)[1].numpy())
            NMI_history.append(NMI)
            
            # Backprop and optimize
            loss = reconst_loss + beta * ( abs(kl_div - C_z) + abs(entropy - C_c) )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose:
                if (i+1) % 100 == 0:
                    print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Entropy: {:.4f}" 
                           .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size,
                                   kl_div.item()/batch_size, entropy.item()/batch_size))
                    print('NMI : ',NMI)
                    logger.write("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Entropy: {:.4f}\n" 
                           .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size,
                                   kl_div.item()/batch_size, entropy.item()/batch_size))
                    logger.write('NMI : {}\n'.format(NMI))
                    rec_loss.append(reconst_loss.item()/batch_size)
                    kl_loss.append(kl_div.item()/batch_size)
            if i>=int(L*ratio):
                break
    logger.close()
    return NMI_history,rec_loss,kl_loss

import warnings
warnings.filterwarnings('ignore') 

# Hyper-parameters
learning_rate = 1e-3
beta = 20
C_z_fin=200
C_c_fin=200

model_G = VAE_Gumbel(z_dim=n_topic).to(device)
optimizer = torch.optim.Adam(model_G.parameters(), lr=learning_rate)

NMI,rec_loss,kl_loss = train_G_modified_loss(model_G, model, train_loader, num_epochs=num_epochs, beta=beta, C_z_fin=C_z_fin, C_c_fin=C_c_fin)


torch.save(model_G.state_dict(), 'save/{}_{}_gumble.model'.format(taskname,n_topic))


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Reconstrurec_lossss')
plt.plot(list(range(len(rec_loss))),rec_loss)

plt.subplot(1,3,2)
plt.title('KL Divergence')
plt.plot(list(range(len(kl_loss))),kl_loss)

plt.subplot(1,3,3)
plt.title('NMI')
plt.plot(list(range(len(NMI))),NMI)

plt.savefig('logs/{}_{}_loss_f-vae-g.png'.format(taskname,n_topic))
#plt.show()


# In[52]:

logger = open('logs/{}_{}_f-vae-g.log'.format(taskname,n_topic),'a',encoding='utf-8')
logger.write('Topics:\n')
for tp in show_topics(model_G,hasLabel=True):
    print(tp)
    logger.write(str(tp)+'\n')



topics_text=show_topics(model_G,hasLabel=True)

cv_coherence_model_ntm = CoherenceModel(topics=topics_text,texts=raw_docs,dictionary=dictionary,coherence='c_v')
cv_coherence_ntm = cv_coherence_model_ntm.get_coherence()
print('Coherence Score (c_v) for f-VAE (Conditional): ',cv_coherence_ntm)
logger.write('Coherence Score (c_v) for f-VAE (Conditional): {}\n'.format(cv_coherence_ntm))

c_uci_coherence_model_ntm = CoherenceModel(topics=topics_text,texts=raw_docs,dictionary=dictionary,coherence='c_uci')
c_uci_coherence_ntm = c_uci_coherence_model_ntm.get_coherence()
print('Coherence Score (c_uci) for f-VAE (Conditional): ',c_uci_coherence_ntm)
logger.write('Coherence Score (c_uci) for f-VAE (Conditional): {}\n'.format(c_uci_coherence_ntm))

c_npmi_coherence_model_ntm = CoherenceModel(topics=topics_text,texts=raw_docs,dictionary=dictionary,coherence='c_npmi')
c_npmi_coherence_ntm = c_npmi_coherence_model_ntm.get_coherence()
print('Coherence Score (c_npmi) for f-VAE (Conditional): ',c_npmi_coherence_ntm)
logger.write('Coherence Score (c_npmi) for f-VAE (Conditional): {}\n'.format(c_npmi_coherence_ntm))

logger.close()


print('Run f-vae-gc')
def train_G_labels(model, preModel, data_loader, num_epochs, beta=1, verbose=True):
    nmi_scores = []
    model.train(True)
    NMI_history = []
    rec_loss,kl_loss = [],[]
    L = len(data_loader)
    for epoch in range(num_epochs):
        
        for i, x in enumerate(data_loader):
            # Forward pass
            x = x.to(device)
            labels = infer_topic_tensor(x,preModel)
            x_reconst, mu, log_var, log_p = model(x)
            p = torch.exp(log_p)
            
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            NMI = normalized_mutual_info_score(labels.cpu().numpy(), p.cpu().max(1)[1].numpy())
            NMI_history.append(NMI)
            
            label_loss = F.nll_loss(log_p, labels)
            
            # Backprop and optimize
            loss = reconst_loss + kl_div + beta * label_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose:
                if (i+1) % 100 == 0:
                    print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Label Loss: {:.4f}" 
                           .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size,
                                   kl_div.item()/batch_size, label_loss/batch_size))
                    print('NMI : ',NMI)
                    rec_loss.append(reconst_loss.item()/batch_size)
                    kl_loss.append(kl_div.item()/batch_size)
            if i>=int(L*ratio):
                break
            
    return NMI_history,rec_loss,kl_loss


n_classes = n_topic
learning_rate = 0.01
model_GC = VAE_Gumbel(z_dim=n_topic).to(device)
optimizer = torch.optim.Adam(model_GC.parameters(), lr=learning_rate)

NMI,rec_loss,kl_loss = train_G_labels(model_GC, model, train_loader, num_epochs=2, beta=10, verbose=True)

torch.save(model_GC.state_dict(), 'save/{}_{}_gc.model'.format(taskname,n_topic))


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Reconstrurec_lossss')
plt.plot(list(range(len(rec_loss))),rec_loss)

plt.subplot(1,3,2)
plt.title('KL Divergence')
plt.plot(list(range(len(kl_loss))),kl_loss)

plt.subplot(1,3,3)
plt.title('NMI')
plt.plot(list(range(len(NMI))),NMI)

plt.savefig('logs/{}_{}_loss_f-vae-gc.png'.format(taskname,n_topic))
#plt.show()


# In[55]:


logger = open('logs/{}_{}_f-vae-gc.log'.format(taskname,n_topic),'a',encoding='utf-8')
logger.write('Topics:\n')
for tp in show_topics(model_GC,hasLabel=True):
    print(tp)
    logger.write(str(tp)+'\n')



topics_text=show_topics(model_GC,hasLabel=True)

cv_coherence_model_ntm = CoherenceModel(topics=topics_text,texts=raw_docs,dictionary=dictionary,coherence='c_v')
cv_coherence_ntm = cv_coherence_model_ntm.get_coherence()
print('Coherence Score (c_v) for f-VAE (Conditional): ',cv_coherence_ntm)
logger.write('Coherence Score (c_v) for f-VAE (Conditional): {}\n'.format(cv_coherence_ntm))

c_uci_coherence_model_ntm = CoherenceModel(topics=topics_text,texts=raw_docs,dictionary=dictionary,coherence='c_uci')
c_uci_coherence_ntm = c_uci_coherence_model_ntm.get_coherence()
print('Coherence Score (c_uci) for f-VAE (Conditional): ',c_uci_coherence_ntm)
logger.write('Coherence Score (c_uci) for f-VAE (Conditional): {}\n'.format(c_uci_coherence_ntm))

c_npmi_coherence_model_ntm = CoherenceModel(topics=topics_text,texts=raw_docs,dictionary=dictionary,coherence='c_npmi')
c_npmi_coherence_ntm = c_npmi_coherence_model_ntm.get_coherence()
print('Coherence Score (c_npmi) for f-VAE (Conditional): ',c_npmi_coherence_ntm)
logger.write('Coherence Score (c_npmi) for f-VAE (Conditional): {}\n'.format(c_npmi_coherence_ntm))

logger.close()

