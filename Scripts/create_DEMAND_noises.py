


import scipy.io.wavfile as wav
import soundfile as sf
import numpy as np
import librosa
import glob
import os

DEMAND_dir = "LibriVAD/Files/Datasets/DEMAND/"
noises_dir = "LibriVAD/Files/Noises/"

def downsample_cafe():
  
  paths = glob.glob(DEMAND_dir + f"SCAFE/*.wav")
  
  print(f"Downsampling Cafe files to 16000 Hz")
  for path in paths:
  
    file = path.split('/')[-1]
    
    sgn, sr = librosa.load(path, sr = 16000)

    sf.write(path, sgn, sr)
    
    
    
def create_DEMAND_noises():

  downsample_cafe()
  
  Categories = ["D","N","O","P","S","T"]
  Categories_full = ["Domestic_noise","Nature_noise","Office_noise",
                     "Public_noise","Street_noise","Transport_noise"]
  
  for j in range(len(Categories)):
    
    category = Categories[j]
        
    paths = glob.glob(DEMAND_dir + f"{category}*/*.wav")
    paths.sort()

    train_paths = []
    val_paths = []
    test_paths = []

    for i in range(len(paths)):
      
      path = paths[i]
      mod = i%16
      
      if mod < 12:
        train_paths.append(path)
      elif mod < 14:
        val_paths.append(path)
      else:
        test_paths.append(path)
    
    train_sgn = []
    for path in train_paths:
      sgn, sr = sf.read(path)
      train_sgn.append(sgn)

    val_sgn = []
    for path in val_paths:
      sgn, sr = sf.read(path)
      val_sgn.append(sgn)
    
    test_sgn = []
    for path in test_paths:
      sgn, sr = sf.read(path)
      test_sgn.append(sgn)
    
    train_sgn = np.hstack(train_sgn)
    val_sgn = np.hstack(val_sgn)    
    test_sgn = np.hstack(test_sgn)
    
    train_sgn = np.int16(train_sgn/np.max(np.abs(train_sgn)) * 32767)
    val_sgn = np.int16(val_sgn/np.max(np.abs(val_sgn)) * 32767)
    test_sgn = np.int16(test_sgn/np.max(np.abs(test_sgn)) * 32767)
    
    os.makedirs(noises_dir + f"{Categories_full[j]}/", exist_ok = True)
    
    wav.write(noises_dir+f"{Categories_full[j]}/{Categories_full[j]}_train.wav",
              train_sgn, sr)
    wav.write(noises_dir+f"{Categories_full[j]}/{Categories_full[j]}_val.wav",
              val_sgn, sr)
    wav.write(noises_dir+f"{Categories_full[j]}/{Categories_full[j]}_test.wav",
              test_sgn, sr)