from __future__ import division
import numpy
import sys, copy
import os
import math
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from scipy.signal import lfilter
import code, copy
from auc_bdnn import enframe 
import torch
import scipy


def mfcc(file_wav, winlen, ovrlen, pre_coef, nfilter, nftt, no_coef):
    #	S. Davis ; P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", IEEE Transactions on Acoustics, Speech, and Signal Processing ( Volume: 28, Issue: 4, Aug 1980 )
    
     fs, speech = speech_wave(file_wav.split(",")[0].strip('\n'))

     eps = numpy.finfo(float).eps
     #
     flen, fsh10 = int(numpy.fix(fs*winlen)),  int(numpy.fix(fs*ovrlen))
     nfr10=int(numpy.floor((len(speech)-(flen-fsh10))/fsh10))


     #for simple  - enegry threshold vad 
     Espeech, foVr = enframe(speech, fs, winlen, ovrlen) #framing (before pre-emphasis)
     Espeech= 20*numpy.log10(numpy.std(Espeech, axis=1, ddof=1)  + eps)
     #====
 
     speech = numpy.append(speech[0],speech[1:]-pre_coef*speech[:-1]) #pre-emphasis
     speech, foVr = enframe(speech, fs, winlen, ovrlen) #framing

     #framing and it's label
     y =  np.load(file_wav.split(",")[1].strip('\n') )
#     y =  (np.loadtxt(os.path.splitext(file_wav.split(",")[1].strip('\n'))[0] + '.txt')).astype(int) #when label in txt format
     y, foVr = enframe(y, fs, winlen, ovrlen)

     #speech/non-speech votting samples
     max_votes = scipy.stats.mode(y, axis=1)
     max_voters_pred = max_votes[0]
     max_voters_pred = np.asmatrix(max_voters_pred).T

     if numpy.size(speech, axis=0) != numpy.size(Espeech, axis=0):
        print("Mismatch frame numbers for  enegry and feature vectors")
     
     if max_voters_pred.shape[0] != y.shape[0]:
         print("Mismatch Label and feature frames")
     

     w = numpy.matrix(numpy.hamming(int(fs*winlen)) )
     w = numpy.tile(w,(numpy.size(speech, axis=0), 1))

     speech = numpy.multiply (speech, w) #apply window


     ff=(fs/2)* (numpy.linspace(0, 1 , int(nftt/2 +1) ))
     fmel=2595*numpy.log10(1+ ff/700) #mel-scale
     fmelmax, fmelmin = numpy.max(fmel), numpy.min(fmel)
     
     filtbankMel= numpy.linspace(fmelmin,fmelmax, nfilter+2) #define filter in mel domain
     filbankF=700*( numpy.power(10, (filtbankMel/2595)) -1)

     #fft
     ffy=numpy.abs(numpy.fft.fft(speech,nftt))     
     ffy, idx =numpy.power(ffy, 2), range(1,int(nftt/2) +1)
     ffy=ffy[:,idx]
 
     BB=(len(ff), nfilter)
     fbank=numpy.zeros(BB)

     for nf in range(0, nfilter):
          fbank[:,nf] = trimf(ff, filbankF[nf], filbankF[nf+1], filbankF[nf+2])
     
     fbank = fbank.T
     fbank = fbank[:,idx].T
     fbnkSum = numpy.matrix(ffy) * numpy.matrix(fbank)  
     
     #dct
     fbankSum_eps = numpy.log10(fbnkSum.T + eps)   
     t=(dct(fbankSum_eps.T, norm = 'ortho')).T
     t= t[1:]  #dicard "c0" #(19, 291)
     t = t[:no_coef,:]

     #append energy
     t = np.vstack((t, Espeech)) #[feat x frames]
     numpy_array = t.T #(frames x feat)

     # d, dd
     #delta -delta
     array_delta = copy.deepcopy(deltas(numpy_array, 2))
     array_delta_delta = copy.deepcopy(deltas(array_delta, 2))
     numpy_array = copy.deepcopy(np.hstack((numpy_array, array_delta, array_delta_delta)))

     return numpy_array[:nfr10,:], max_voters_pred[:nfr10,:]
    


def speech_wave(fileName_):

     (fs,sig) = wav.read(fileName_)
     sig=sig/numpy.amax(numpy.abs(sig)) #normalize the signal (for the feature extraction)
     return fs, sig
 

def trimf(Xx, aA_, bB_, cC_):
    
    if aA_ > bB_:
          print("Parameter: a > b")
          exit()
    elif  bB_ > cC_:
          print("Parameter: b > c")
          exit()
    elif  aA_ > cC_:
          print("Parameter: a > c")
          exit()
    
    BB=len(Xx) ## ff
    ky = numpy.zeros(BB)

    index=numpy.where( (Xx <= aA_) | (cC_<= Xx))     
    ky[index] = numpy.zeros(len(index[0]))            

    # slope 1
    if aA_ != bB_:
       index = numpy.where((aA_ < Xx) & (Xx < bB_))
       ky[index] = (Xx[index]- aA_)/(bB_ - aA_)

    # slope 2
    if bB_ != cC_:
       index = numpy.where( (bB_ < Xx) & (Xx < cC_))
       ky[index] = (cC_ - Xx[index])/(cC_ - bB_)


    #Center 
    index = numpy.where(Xx == bB_)
    ky[index] = numpy.ones(len(index[0]))  
    
    return ky  
 

def deltas(feat, N): 
        if N < 1:
                raise ValueError('N must be an integer >= 1')
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        NUMFRAMES = len(feat)
        delta_feat = np.empty_like(feat)
        padded = np.concatenate((np.tile(feat[0,:],(N,1)), feat, np.tile(feat[-1,:],(N,1))), axis=0)
        for t in range(NUMFRAMES):
                delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1])
        delta_feat /= denominator
        return delta_feat

def cmvn(numpy_array):
     #zero-mean std      
     _mean = np.mean(numpy_array, axis=0, dtype='float32')
     _std = np.std(numpy_array, axis=0, ddof=1, dtype='float32')
     # if any element in _std value "0" replace by 1
     _std = np.where(_std==0, 1, _std)
     Z = (numpy_array - _mean)/_std

     return Z




if __name__ == '__main__':
   winlen, ovrlen, pre_coef, nfilter, nftt, no_coef = 0.025, 0.01, 0.97, 24, 512, 12
   #[window size (sec)], [frame shift(sec)], [pre-emp coeff],

   scp = np.genfromtxt(sys.argv[1],dtype='str')
   for x in scp:
     #print(x)  
     feat, Label = mfcc(x, winlen, ovrlen, pre_coef, nfilter, nftt, no_coef)
     feat = cmvn (feat) # zero-mean unit variance normalization
     feat  = torch.FloatTensor(feat)
     Label = torch.FloatTensor(Label)
     #write in torch format
     torch.save(feat, x.split(",")[2].strip('\n')+".pt")
     torch.save(Label, x.split(",")[2].strip('\n')+".lab.pt")




