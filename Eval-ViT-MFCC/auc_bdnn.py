from __future__ import division, print_function, unicode_literals
import scipy
from sklearn import metrics
import numpy as np
import scipy.io.wavfile as wav
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os, sys, code, datetime
import shutil
import struct, copy


def uW_context(X_batch, w, u, bDnnWin):
    if X_batch.shape[0] < 2:
       raise Exception("data is not matrix or having one frame only")

    index=np.arange(X_batch.shape[0]) #[frame x feat]
    Xarray = copy.deepcopy(X_batch) # [0] current location

    #generate left context   #w=19, u=9
    L1 = np.sort(np.concatenate((np.arange(start=-1-u,stop=-w, step=-u), np.array([-w]))))
    L = np.sort( np.unique(np.concatenate((L1, np.array([-1])))) ) #array([-19, -10,  -1])

    #right context
    R1 = np.concatenate((np.array([1]), np.arange(start=1+u,stop=w, step=u))) #
    R = np.sort( np.unique(np.concatenate((R1, np.array([w])))) ) #array([1, 10, 19])

    #both L and R are symmetric
    for i in np.arange(len(L)-1,-1,-1): #negative indices
         x = np.abs(L[i])
         t=np.pad(index[:-x],pad_width=(len(index) - len(index[:-x]),0), mode='constant', constant_values=index[0])
         Xarray = np.hstack((X_batch[t,:], Xarray))

    for i in np.arange(0, len(R)): # positive indic
         x = np.abs(R[i])
         t = np.pad(index[x:],pad_width=(0, len(index) - len(index[x:])), mode='constant', constant_values=index[-1])
         Xarray=np.hstack((Xarray,X_batch[t,:]))

    Xarray = np.sort(Xarray)

    if bDnnWin*X_batch.shape[1] != Xarray.shape[1]:
       raise Exception("mismatch contextual bDnnWin dimension")

    return Xarray


def LRcontext(X_batch, Lt, Rt, bDnnWin):
    if X_batch.shape[0] < 2:
       raise Exception("data is not matrix or having one frame only")

    index=np.arange(X_batch.shape[0]) #[frame x feat]
    Xarray = copy.deepcopy(X_batch) # [0] current location


    #generate left context   #w=19, u=9
    L1 = np.sort(np.concatenate((np.arange(start=Lt,stop=0, step=1), np.array([Lt]))))
    L = np.sort( np.unique(np.concatenate((L1, np.array([Lt])))) )

    #right context
    R1 = np.concatenate((np.array([1]), np.arange(start=1,stop=Rt, step=1))) #
    R = np.sort( np.unique(np.concatenate((R1, np.array([Rt])))) )

    #both L and R are symmetric
    for i in np.arange(len(L)-1,-1, -1): #negative indices (read like that -1, -2, ,,,)
         x = np.abs(L[i])
         t=np.sort(np.pad(index[:-x],pad_width=(len(index) - len(index[:-x]),0), mode='constant', constant_values=index[0]))
         Xarray = np.hstack((X_batch[t,:], Xarray))

    for i in np.arange(0, len(R)): # positive indic
         x = np.abs(R[i])
         t = np.pad(index[x:],pad_width=(0, len(index) - len(index[x:])), mode='constant', constant_values=index[-1])
         Xarray=np.hstack((Xarray,X_batch[t,:]))

    if bDnnWin*X_batch.shape[1] != Xarray.shape[1]:
       raise Exception("mismatch contextual bDnnWin dimension")

    return Xarray



def enframe(speech, fs, winlen, ovrlen):
     #split the speech data into frames  
     N, flth, foVr = len(speech), int(np.fix(fs*winlen)),  int(np.fix(fs*ovrlen))

     if len(speech) < flth:
        print("speech file length shorter than 1-frame")
        exit()

     frames = int(np.ceil( (N - flth )/foVr)) + 1
     slen = (frames-1)*foVr + flth
     if len(speech) < slen:
        signal = np.concatenate((speech, np.zeros((slen - N))))

     else:
        signal = copy.deepcopy(speech)

     idx = np.tile(np.arange(0,flth),(frames,1)) + np.tile(np.arange(0,(frames)*foVr,foVr),(flth,1)).T
     idx = np.array(idx,dtype=np.int64)

     return signal[idx], foVr


def auc_acc(tstscp, model, device, tw, tu, bDnnWin, cutThres):
   score =[]
   Lab = []
   for fe in tstscp:
      try:
         sgn = torch.load(fe.split(",")[2].strip('\n')+".pt", weights_only=True) 
         sgn = sgn.detach().cpu().numpy()
         sgn =  copy.deepcopy( uW_context(sgn, tw, tu, bDnnWin))
         sgn = torch.from_numpy(sgn)

         max_voters_pred = torch.load(fe.split(",")[2].strip('\n')+".lab.pt", weights_only=True)
         max_voters_pred = max_voters_pred.detach().cpu().numpy()

         if sgn.shape[0] == max_voters_pred.shape[0]:
            sgn = sgn.to(device)
            sgn = sgn.unsqueeze(0)
            bn1, bn2, logit = model(sgn.float())
            logit = logit.squeeze(0)
            conTxt = logit.shape[1]
            if conTxt != bDnnWin:
                raise Exception("mismatch score and frame context bDnnWin")
            logit = np.asmatrix(np.sum(logit.cpu().detach().numpy(), 1)/conTxt)
            logit = logit.T
            logit = np.concatenate((1-logit, logit), axis = 1)

            if logit.shape[0] == max_voters_pred.shape[0]:
               score.append(logit)
               Lab.append(np.asmatrix(max_voters_pred))

      except ValueError:
           pass
   score = np.squeeze(np.concatenate(score, axis=0 ))
   Lab = np.squeeze(np.concatenate(Lab, axis=0 ))
   Lab = Lab.T
   tot=0
   corr=0
   ACC1=0
   if cutThres == 'True':
        fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(np.asarray(Lab)), np.asarray(score[:,1]), pos_label=1)
        muc1 = metrics.auc(fpr, tpr)
        pred = np.argmax(score, axis=1) #ML
        if pred.shape[0] == Lab.shape[0]:
           pred = pred - Lab
           pred = np.array(np.where(pred == 0))
           corr =  pred.shape[1]
           tot =  score.shape[0]
           ACC1 = corr/tot

   else:
        pred = [1 if  score >= cutThres else 0 for score in np.asarray(score[:,1])]
        muc1 = roc_auc_score(np.squeeze(np.asarray(Lab)), pred)
        pred = np.matrix(np.array(pred)).T
        fpr, tpr = 0, 0
        if pred.shape[0] == Lab.shape[0]:
           pred = pred - Lab
           pred = np.array(np.where(pred == 0))
           corr =  pred.shape[1]
           tot =  score.shape[0]
           ACC1 = corr/tot


   muc1 =round(muc1, 4)
   if cutThres == 'True':
             return muc1, round(ACC1,4), score.shape[0], thresholds, fpr, tpr
   else:
             return muc1, round(ACC1,4), score.shape[0], cutThres, fpr, tpr



def auc_acc_bcres(tstscp, model, device, tw, tu, bDnnWin, featdim, cutThres):
   score =[]
   Lab = []
   for fe in tstscp:
      try:
         sgn = torch.load(fe.split(",")[2].strip('\n')+".pt", weights_only=True)
         sgn = sgn.detach().cpu().numpy()
         sgn =  copy.deepcopy(LRcontext(sgn, tw, tu, bDnnWin)) #context
         sgn = torch.from_numpy(sgn)

         max_voters_pred = torch.load(fe.split(",")[2].strip('\n')+".lab.pt", weights_only=True)
         max_voters_pred = max_voters_pred.detach().cpu().numpy()

         if sgn.shape[0] == max_voters_pred.shape[0]:
            sgn = sgn.to(device)
            sgn = torch.reshape(sgn, [sgn.shape[0], -1, featdim])
            sgn = sgn.unsqueeze(0) #[1, 931, 11, 39]
            sgn = sgn.permute(1,0,3, 2)
            logit = model(sgn.float()) #[BZ, 1, 40, #Frames]
            logit = torch.softmax(logit, dim=1)
            logit = logit.cpu().detach().numpy()
            logit = np.asmatrix(logit)
            score.append(logit)
            Lab.append(np.asmatrix(max_voters_pred))
    
      except ValueError:
           pass
      score = np.squeeze(np.concatenate(score, axis=0 ))
      Lab = np.squeeze(np.concatenate(Lab, axis=0 ))
      Lab = np.squeeze(np.asarray(Lab))

      if cutThres == 'True':
          fpr, tpr, thresholds = metrics.roc_curve(Lab, np.asarray(score[:,1]), pos_label=1)
          muc1 = metrics.auc(fpr, tpr)
          pred = np.argmax(score, axis=1) #ML
          if pred.shape[0] == Lab.shape[0]:
              ACC1 = accuracy_score(Lab, np.asarray(pred))
      else:
          pred = [1 if  score >= cutThres else 0 for score in np.asarray(score[:,1])]
          pred = np.squeeze(np.asarray(pred))
          muc1 = roc_auc_score(Lab, pred)
          if pred.shape[0] == Lab.shape[0]:
              ACC1 = accuracy_score(Lab, pred)

      muc1 =round(muc1, 4)
      if cutThres == 'True':
           return muc1, round(ACC1,4), score.shape[0], thresholds
      else:
           return muc1, round(ACC1,4), score.shape[0], cutThres
              


    
