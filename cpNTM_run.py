import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.cluster import normalized_mutual_info_score
from data import *
from utils import *

parser = argparse.ArgumentParser(description='Neural Topic Model')

# 添加参数步骤
parser.add_argument('--taskname', type=str,default='sub',help='Name of the task e.g subX')
parser.add_argument('--n_topic', type=int,default=20,help='Number of the topics')
parser.add_argument('--num_epochs',type=int,default=2,help='Num of epochs')
parser.add_argument('--batch_size',type=int,default=128,help='Batch Size')
parser.add_argument('--gpu',type=str,default='0',help='GPU device e.g 1')
parser.add_argument('--ratio',type=float,default=1.0,help='Ratio of the train data for actual use')
parser.add_argument('--use_stopwords',type=bool,default=True,help='Whether to use stopwords or not')
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu

n_topic = args.n_topic
taskname = args.taskname
num_epochs = args.num_epochs
batch_size = args.batch_size
ratio = args.ratio
use_stopwords = args.use_stopwords


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# VAE model
class VAE(nn.Module):
    def __init__(self, bow_size=50000, h_dim=4096, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(bow_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, z_dim)
        self.fc5 = nn.Linear(z_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, bow_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, log_var = self.fc2(h), self.fc3(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        theta = torch.softmax(self.fc4(z),dim=1)
        #theta = torch.softmax(z,dim=1)
        return theta

    def decode(self, theta):
        h = F.relu(self.fc5(theta))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        theta = self.reparameterize(mu, log_var)
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var


# Here for the loss, instead of MSE for the reconstruction loss, we take BCE. The code below is still from the pytorch tutorial (with minor modifications to avoid warnings!).


print('Loading train_data ...')
train_loader,vocab = get_batch(taskname,use_stopwords,batch_size)

# Hyper-parameters
learning_rate = 1e-3

model = VAE(bow_size=len(vocab),h_dim=1024,z_dim=n_topic).to(device)
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print('Start training...')
def train(model, data_loader, num_epochs):
    rec_loss = []
    kl_loss = []
    logger = open('logs/{}_{}_vae.log'.format(taskname,n_topic),'w',encoding='utf-8')
    logger.write('='*30+'vae'+'='*30+'\n')
    L = len(data_loader)
    for epoch in range(num_epochs):
        for i, x in enumerate(data_loader):
            # Forward pass
            x = x.to(device)
            x_reconst, mu, log_var = model(x)

            # Compute reconstruction loss and kl divergence
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Backprop and optimize
            loss = reconst_loss + kl_div
            #loss = reconst_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size, kl_div.item()/batch_size))
                logger.write("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}\n" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size, kl_div.item()/batch_size))
                rec_loss.append(reconst_loss.item()/batch_size)
                kl_loss.append(kl_div.item()/batch_size)
            if i>=int(L*ratio):
                break
    logger.write('='*60+'\n\n')
    logger.close()
    return rec_loss,kl_loss


rec_loss,kl_loss = train(model ,train_loader, num_epochs=num_epochs)

torch.save(model.state_dict(), 'ckpt/{}_{}.model'.format(taskname,n_topic))


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title('Reconstrurec_loss for {} {} topics'.format(taskname,n_topic))
plt.plot(list(range(len(rec_loss))),rec_loss)

plt.subplot(1,2,2)
plt.title('KL Divergence for {} {} topics'.format(taskname,n_topic))
plt.plot(list(range(len(kl_loss))),kl_loss)
plt.savefig('logs/{}_{}.png'.format(taskname,n_topic))


import jieba

def infer_topic(strText,model):
    bow = doc2bow(list(jieba.cut(strText)))
    bow = bow.unsqueeze(0)
    return model.encode(bow.cuda())[0].argmax(1)

def infer_topic_tensor(bow,model):
    return model.encode(bow.cuda())[0].argmax(1)

#infer_topic('你或许需要做个核磁共振',model)
#infer_topic_tensor(next(iter(train_loader)),model)



def show_topics(model=model, n_topic=n_topic, topK=20, showWght=False,fix_topic=None,hasLabel=False):
    global vocab
    if isinstance(model,nn.DataParallel):
        model = model.module
    topics = []
    idxes = torch.eye(n_topic).cuda()
    if hasLabel:
        word_dists = model.decode(idxes,idxes)
    else:
        word_dists = model.decode(idxes)
    vals,indices = torch.topk(word_dists,topK,dim=1)
    vals = vals.cpu().tolist()
    indices = indices.cpu().tolist()
    if fix_topic is None:
        for i in range(n_topic):
            if showWght==True:
                topics.append([(wght,vocab.idx2word[idx]) for wght,idx in zip(vals[i],indices[i])])
            else:
                topics.append([vocab.idx2word[idx] for idx in indices[i]])
    else:
        if showWght==True:
            topics.append([(wght,vocab.idx2word[idx]) for wght,idx in zip(vals[fix_topic],indices[fix_topic])])
        else:
            topics.append([vocab.idx2word[idx] for idx in indices[fix_topic]])
    return topics

logger = open('logs/{}_{}_vae.log'.format(taskname,n_topic),'a',encoding='utf-8')
logger.write('Topics:\n')
for tp in show_topics():
    print(tp)
    logger.write(str(tp)+'\n')
exit(0)


#Evaluate the Model with Topic Coherence
from gensim.models import CoherenceModel

topics_text=show_topics()

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
