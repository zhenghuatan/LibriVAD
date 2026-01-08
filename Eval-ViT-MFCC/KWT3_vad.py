# The ViT part of the code is adapted from: https://github.com/lucidrains/vit-pytorch
# The original ViT code Copyright (c) 2020 Phil Wang
# Licensed under the MIT License
 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys, os
import code
import datetime
import math
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import LambdaLR
import torch.fft
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from auc_Vit import split_data_into_sequences, auc_acc
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(42)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, pre_norm=True, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                P_Norm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                P_Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class KWT(nn.Module):
    def __init__(self, input_res, patch_res, num_classes, dim, depth, heads, mlp_dim, channels, dim_head, dropout, emb_dropout, pre_norm = True, **kwargs):
        super().__init__()
        
        self.num_patches = int(input_res[0]/patch_res[0] * input_res[1]/patch_res[1])
        
        self.patch_dim = channels * patch_res[0] * patch_res[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res[0], p2 = patch_res[1]),
            nn.Linear(self.patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, pre_norm, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) 
        
    def forward(self, x): #[512, 100, 39]
        x = x.permute(0,2,1) #> [512, 39, 100]
        x = x.unsqueeze(1) #> [512, 1, 39, 100]
        x = self.to_patch_embedding(x) #> [512, 100, 192]
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n ]
        x = self.dropout(x)

        x = self.transformer(x) #> [512, 100, 192]
        x = self.to_latent(x)
        x = x.reshape(-1, x.shape[2]) #> [51200, 192])
        return self.mlp_head(x)


class WarmupCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=0.0, last_epoch=-1):
        """
        Initialize the WarmupCosineAnnealingLR scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
            T_max (int): Total number of epochs for the cosine annealing schedule.
            warmup_epochs (int): Number of epochs for the linear warmup phase.
            eta_min (float): Minimum learning rate after annealing.
            last_epoch (int): The index of the last epoch when resuming training.
        """
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (self.T_max - self.warmup_epochs)
                progress_tensor = torch.tensor(progress, dtype=torch.float32)  # Convert to tensor with float32 dtype
                return self.eta_min + 0.5 * (1 - self.eta_min) * (1 + torch.cos(progress_tensor * torch.pi))
        
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, lr_lambda, last_epoch)


def myfun(fe):
    try:
        feat = torch.load(fe.split(",")[2].strip('\n')+".pt", weights_only=True)
        label = torch.load(fe.split(",")[2].strip('\n')+".lab.pt", weights_only=True)
        feat = feat.detach().cpu().numpy() #[98, 39]
        label = label.detach().cpu().numpy()
        return (feat, label)              
    except:
        return('None','None') 

def readBatchScp(batch_scp, num_cores, sequence_length):
    feat = []
    y_train = []
    X_batch = Parallel(n_jobs=num_cores)(delayed(myfun)(batch_scp[i] )for i in range(len(batch_scp)))    
    for t1, t2 in X_batch:
        try:
               t1_array = np.array(t1, dtype=float)  # Ensure elements are numeric
               t2_value = t2  # Label is numeric or None
               if (t2_value is not None  and not np.any(np.isnan(t1_array)) and not np.any(np.isinf(t1_array)) and (not np.isnan(t2_value) if isinstance(t2_value, (int, float)) else True)  and (not np.isinf(t2_value) if isinstance(t2_value, (int, float)) else True)):
                   X_sequences, y_sequences, t3, t3_seq = split_data_into_sequences(t1, t2, sequence_length)
                   feat.append(X_sequences)
                   y_train.append(y_sequences)
        except: 
              feat_array = np.array(feat, dtype=float)  # Ensure elements are convertible to float
              y_train_array = np.array(y_train, dtype=float)

    feat = np.concatenate(feat, axis=0)  
    y_train = np.concatenate(y_train, axis=0)

    if len(feat) == len(y_train):
        feat = torch.from_numpy(feat)
        y_train = torch.from_numpy(y_train)

        return feat, y_train


class CustomDataset_train(Dataset):
    def __init__(self, time_series, labels):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        time_serie = self.time_series[idx]  # .clone().detach()
        label = self.labels[idx]  # label

        return (time_serie, label)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def predict(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir='checkpoints', max_checkpoints=5):
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create checkpoint filename
    checkpoint_filename = str(checkpoint_dir) + '/' + 'checkpoint_epoch_' + str(epoch) + '.pth'

    # Save the checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_filename)
    print("Checkpoint saved at", checkpoint_filename)

    # Get list of all checkpoint files in the directory
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )

    # Remove older checkpoints if the number exceeds the max limit
    if len(checkpoint_files) > max_checkpoints:
        for old_checkpoint in checkpoint_files[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print("Removed old checkpoint:", old_checkpoint)

def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints'):
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )

    if not checkpoint_files:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0, None

    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("Checkpoint loaded from", checkpoint_path, ": Resuming from epoch", epoch)
    return model, optimizer, epoch, loss


# Hyperparameters
mlp_dim = 768   # projected output dimension in FF layer
num_classes = 2 # number of class
dropout = 0.1 
batch_size = 32
depth = 12         #number of encoder layer in transformer
heads = 3          #number of head in Multi-head attension
dim = 192            #  linear transformation nn.Linear(..., dim) of patch dimension (embedding)
pool = 'cls'
channels = 1
dim_head = dim // heads  # multi-head dimension
dropout = 0.1
emb_dropout = 0.1 
pre_norm = 'False'


learning_rate = 0.001
weight_decay = 0.1
label_smoothing = 0.1
T_max = 1000  # Total number of iterations for one cycle
eta_min = 0  # Minimum learning rate
sequence_length = 100
input_feat = [39, sequence_length]
patch_size = [39, 1]
num_cores = 10

model = KWT(input_feat, patch_size , num_classes, dim, depth, heads, mlp_dim, channels , dim_head, dropout, emb_dropout, pre_norm)
model.to(device)
#print(model)

# Define loss function and optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#scheduler =optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of

scheduler = WarmupCosineAnnealingLR(optimizer, T_max=100, warmup_epochs=10, eta_min=0.0001)
criterion = nn.CrossEntropyLoss()


phase = open('phase.txt')
content= phase.read()
Trn=content.rstrip()

fpath=open('dir.txt')
content= fpath.read()
model_dir=content.rstrip()

if not os.path.exists(model_dir):
   os.makedirs(model_dir)



n_epoch = int(np.loadtxt('epoch.txt'))


if  str(Trn) == "Train":
    trnScp = np.genfromtxt('DNN.trn.scp', dtype='str')
    print('number of training files: ', trnScp.shape[0])
    uBatch_size = 10000
    fname = model_dir  + '/training_loss.txt'
    start_epoch = 0
    total_epochs = n_epoch
    model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, model_dir)
    if start_epoch ==0:
       fp = open(fname,"w")
    else:
       fp = open(fname,"a+")

    for epoch in range(0, n_epoch):
        model.train()
        # shuffle the data
        idx = np.arange(0, len(trnScp))
        np.random.shuffle(idx)
        et = datetime.datetime.now()
        trainingLoss = []

        for i in range(0, len(trnScp), uBatch_size):  # utterance at a time (say)/batch
                  newidx = idx[i:i+uBatch_size]
                  # get files for the index of current batch
                  batch_scp = trnScp[newidx]
                  X_batch, y_train = readBatchScp(batch_scp, num_cores, sequence_length)
                  train_dataset = CustomDataset_train(X_batch, y_train)
                  trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                  for batch, labels in trainloader:
                      optimizer.zero_grad()
                      if batch.shape[0] > 1:
                         labels = np.array(labels, int)
                         labels = torch.from_numpy(labels).to(device)
                         batch = batch.to(device)
                         output = model(batch)
                         loss = criterion(output, labels.reshape(-1,1).squeeze(1)) #([51200]
                         loss.backward()
                         optimizer.step()
                         trainingLoss.append(loss.detach().cpu().numpy())
        scheduler.step()
        torch.save(model.state_dict(), open(os.path.join(model_dir + '/model_epoch_%d' % (epoch + 1) + '.model'), 'wb'))
        # Save checkpoint after each epoch
        save_checkpoint(epoch + 1, model, optimizer, np.mean(trainingLoss), model_dir, 1)
        # Dev set
        model.eval()
        noises = [ "SSN_noise",  "Domestic_noise", "Nature_noise",  "Office_noise", "Public_noise", "Street_noise",  "Transport_noise",  "Babble_noise", "City_noise"]
        snr = [ "-5", "0", "5", "10", "15", "20"]
        devscp = []
        for nx in noises:
           for db in snr:
               tmp = np.genfromtxt('dev-clean/' + str(nx) + '_'+str(db) + '.lst',dtype='str')
               devscp.append(tmp)
        tmp = np.genfromtxt('dev-clean/clean_clean.lst',dtype='str')
        devscp.append(tmp)
        devscp = np.asmatrix(np.squeeze(np.concatenate(devscp, axis=0 )))
        devLoss = []
        for i in range(0, len(devscp), uBatch_size):  # utterance at a time (say)/batch
                  newidx = idx[i:i+uBatch_size]
                  # get files for the index of current batch
                  batch_scp = trnScp[newidx]
                  X_batch, y_train = readBatchScp(batch_scp, num_cores, sequence_length)
                  dev_dataset = CustomDataset_train(X_batch, y_train)
                  devloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                  for batch, labels in devloader:
                      optimizer.zero_grad()
                      if batch.shape[0] > 1:
                         labels = np.array(labels, int)
                         labels = torch.from_numpy(labels).to(device)
                         batch = batch.to(device)
                         output = model(batch)
                         devloss = criterion(output, labels.reshape(-1,1).squeeze(1)) #([51200]
                         devLoss.append(devloss.detach().cpu().numpy())
        print("%s epoch  %d --bsize %d -- --train-loss %.6f --dev-loss %.6f" %(et, epoch, batch_size,np.mean(trainingLoss), np.mean(devLoss)))
        fp.write("%s epoch  %d --bsize %d -- --train-loss %.6f --dev-loss %.6f\n" %(et, epoch, batch_size, np.mean(trainingLoss), np.mean(devLoss)))
        fp.flush()
    fp.close()

elif  str(Trn) == "Test":
     #eval res file-name
     fpath1=open('EvalFile.txt')
     content= fpath1.read()
     EvalRes=content.rstrip()

     print('model --> %s' %(model_dir))
     PATH= model_dir + '/' + 'model_epoch_' + str(n_epoch) +'.model'
     try:
         model.load_state_dict(torch.load(PATH))
     except:
         model.load_state_dict(torch.load(PATH, map_location="cpu"))
         model.eval()

     auc_score = []
     acc_score = []
     noises = [ "SSN_noise",  "Domestic_noise", "Nature_noise",  "Office_noise", "Public_noise", "Street_noise",  "Transport_noise",  "Babble_noise", "City_noise"]
     snr = [ "-5", "0", "5", "10", "15", "20"]
     header = []
     temp_score =[]
     tmp_acc = []
     print('Evalset scoring...')
     finename = model_dir + '/' + str(EvalRes) + '.testdata'
     finenameFrm = open(model_dir + '/' + str(EvalRes) + '.testdata.frm', 'w')
     #Eval-data(noise dependent)
     for nx in noises:
         for db in snr:
             header.append([str(nx) + ',' + str(db)])
             tstscp =np.genfromtxt('test-clean/' + str(nx) + '_'+str(db) + '.lst',dtype='str')
             muc1, ACC1, ndataP, thr,  fpr, tpr, score, label = auc_acc(tstscp, model, device, sequence_length, 'True')
             finenameFrm.write("%s %s  %d\n" %(nx,db,score.shape[0])) 
             finenameFrm.flush()
             temp_score.append([muc1])
             tmp_acc.append([ACC1])
     #clean data
     tstscp =np.genfromtxt('test-clean/clean_clean.lst',dtype='str')
     muc1, ACC1, ndataP, thr,  fpr, tpr, score, label = auc_acc(tstscp, model,  device, sequence_length, 'True')
     temp_score.append([muc1])
     tmp_acc.append([ACC1])
     header.append(['clean'+','+'clean'])

     temp_score = np.asmatrix(np.squeeze(np.concatenate(temp_score, axis=0 ))).T
     tmp_acc = np.asmatrix(np.squeeze(np.concatenate(tmp_acc, axis=0 ))).T

     header = np.squeeze(np.concatenate(header, axis=0 ))
     text_array = header.reshape(-1, 1).astype(object)
     combine = np.hstack([text_array, temp_score])
     np.savetxt(str(finename)+'.auc.csv', combine, delimiter=',', fmt='%s')

     combine = np.hstack([text_array, tmp_acc])
     np.savetxt(str(finename)+'.acc.csv', combine, delimiter=',', fmt='%s')
     finenameFrm.close()

#     print('Devset scoring...')
     #development-set
#     header = []
#     temp_score =[]
#     tmp_acc = []
#     finename = model_dir + '/' + str(EvalRes)+ '.devdata'
#     finenameFrm = open(model_dir + '/' + str(EvalRes) + '.devdata.frm', 'w')
#     for nx in noises:
#         for db in snr:
#             header.append([str(nx) + ',' + str(db)])
#             tstscp =np.genfromtxt('dev-clean/' + str(nx) + '_'+str(db) + '.lst',dtype='str')
#             muc1, ACC1, ndataP, thr,  fpr, tpr, score, label = auc_acc(tstscp, model, device, sequence_length, 'True')
#             finenameFrm.write("%s %s  %d\n" %(nx,db,score.shape[0]))
#             finenameFrm.flush()
#             temp_score.append([muc1])
#             tmp_acc.append([ACC1])
     #clean data
#     tstscp =np.genfromtxt('dev-clean/clean_clean.lst',dtype='str')
#     muc1, ACC1, ndataP, thr,  fpr, tpr,score, label = auc_acc(tstscp, model, device, sequence_length, 'True')
#     temp_score.append([muc1])
#     tmp_acc.append([ACC1])
#     header.append(['clean'+','+'clean'])

#     temp_score = np.asmatrix(np.squeeze(np.concatenate(temp_score, axis=0 ))).T
#     tmp_acc = np.asmatrix(np.squeeze(np.concatenate(tmp_acc, axis=0 ))).T

#     header = np.squeeze(np.concatenate(header, axis=0 ))
#     text_array = header.reshape(-1, 1).astype(object)
#     combine = np.hstack([text_array, temp_score])
#     np.savetxt(str(finename)+'.auc.csv', combine, delimiter=',', fmt='%s')

#     combine = np.hstack([text_array, tmp_acc])
#     np.savetxt(str(finename)+'.acc.csv', combine, delimiter=',', fmt='%s')
#     finenameFrm.close()
elif str(Trn) == "Test_ind":
     #eval res file-name
     fpath1=open('EvalFile.txt')
     content= fpath1.read()
     EvalRes=model_dir + '/' + content.rstrip()

     print('model --> %s' %(model_dir))
     PATH= model_dir + '/' + 'model_epoch_' + str(n_epoch) +'.model'
     try:
         model.load_state_dict(torch.load(PATH))
     except:
         model.load_state_dict(torch.load(PATH, map_location="cpu"))
     model.eval()

     noises = [ "SSN_noise",  "Domestic_noise", "Nature_noise",  "Office_noise", "Public_noise", "Street_noise",  "Transport_noise",  "Babble_noise", "City_noise"]
     snr = [ "-5", "0", "5", "10", "15", "20"]
     tstscp = []
     for db in snr:
         for nx in noises: #all noises
             tstscp.append(np.genfromtxt('test-clean/' + str(nx) + '_'+str(db) + '.lst',dtype='str'))
     tstscp =  np.squeeze(np.concatenate(tstscp, axis=0 ))
     print(tstscp)

     finename = str(EvalRes) + '.testdata'
     #no clean
     muc1, ACC1, ndataP, xyz, fpr, tpr, score, label = auc_acc(tstscp, model, device, sequence_length, 'True') #thr)
     np.savetxt(str(finename) + '.score', score, fmt='%.4f')
     np.savetxt(str(finename) + '.label', label, fmt='%.4f')
 
elif str(Trn) == "Test_voices":  
     #eval res file-name
     fpath1=open('EvalFile.txt')
     content= fpath1.read()
     EvalRes=content.rstrip()

     print('model --> %s' %(model_dir))
     PATH= model_dir + '/' + 'model_epoch_' + str(n_epoch) +'.model'
     try:
         model.load_state_dict(torch.load(PATH))
     except:
         model.load_state_dict(torch.load(PATH, map_location="cpu"))
     model.eval()

     finename = model_dir + '/' + str(EvalRes)+ '.voices'
     fp = open(finename,"w")
     room = [ "rm1", "rm2", "rm3", "rm4"]
     mic = [ "mc01", "mc05" ]
     dist = [ "clo", "far" ]
     distract = [ "babb", "musi", "none", "tele" ]
     for rm in room:
         for mik in mic: 
             for distant in dist:
                 for distr in distract:
                     try:
                        print([str(rm) + '_' + str(mik) + '_' + str(distant) + '_' + str(distr)])
                        tstscp = np.genfromtxt('list_voices/' + str(rm) + '_' + str(mik) + '_' + str(distant) + '_' + str(distr) + '.lst',dtype='str')
                        muc1, ACC1, ndataP, xyz, fpr, tpr, score, label = auc_acc(tstscp, model, device, sequence_length, 'True') #thr)
                        fp.write("%s, %d, %s, %f\n" %([str(rm) + '_' + str(mik) + '_' + str(distant) + '_' + str(distr)], len(tstscp), 'muc', muc1))
                        fp.write("%s, %d, %s %f\n" %([str(rm) + '_' + str(mik) + '_' + str(distant) + '_' + str(distr)],len(tstscp), 'acc', ACC1))
                        fp.flush()
                     except: 
                        file_path = 'list_voices/' + str(rm) + '_' + str(mik) + '_' + str(distant) + '_' + str(distr) + '.lst'
                        if not os.path.exists(file_path):
                            print("File does not exist:", str(file_path))
                        elif os.path.getsize(file_path) == 0:
                            print("File exists but is empty:", str(file_path))
     #
     fp.close()


