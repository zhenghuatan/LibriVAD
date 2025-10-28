# This script was developed in order to generate the city noise signal.
# Due to the large size of the WHAM! noise dataset, it was decided to reduce 
# the number of files that contribute to the city noise used in this project

import scipy.io.wavfile as wav
import numpy as np
import os  
import sys 
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pydub import AudioSegment

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf



def categorize(df):
  """
  Utilizing WHAM! metadata to split them into sub-dataframes based on the 
  recording location
  
  Input:
    - df     : The dataframe containing the metadata
  
  Output:
    - subsets: A list of numpy arrays each containing the file paths of a 
               unique location    
  """
  
  subsets = [np.array(group['utterance_id'])
              for _, group in df.groupby('Location ID')]

  return subsets






def extract_info(file):

  """
  Function for reading, scaling and returning the audio file, along with its 
  length and title. This is used for accelerating the generation process.
  
  Input:
    - file          : The path to the audio file (same for output)
  
  Outputs:
    - file
    - scaled_signal : The scaled audio waveform (to 16-bit)
    - sgn_len       : The length of the signal
               
  """
  
  sr, sgn = wav.read(file)
  
  scaled_signal = lf.scale_signal(sgn)
  
  sgn_len = len(sgn) 
  
  return file, scaled_signal, sgn_len






def process_subsets(df, wham_folder):
  
  """
  Utilizing the categorize function, the extract_info function and threading,
  to accelerate the processing of the files.
  
  Inputs:
    - df               : The dataframe containing the metadata
    - wham_folder      : The path to the WHAM noise folder
  
  Output:
    - processed_subsets: A list of dictionaries containing the audio files and  
                         other relevant information regarding the noise  
                         generation process
  """
  
  subsets = categorize(df)
  processed_subsets = []
  m = len(subsets)
  j = 0
  
  for subset in subsets:
    j += 1
    
    n = len(subset)
    processed_subset = {}
    
    # Utilising asynchronous execution with threads for the files
    with ThreadPoolExecutor() as executor:
    
      futures = {executor.submit(extract_info, os.path.join(wham_folder, file)): file for file in subset}
      
      for i, future in enumerate(as_completed(futures)):

        try:
            file, signal, length = future.result()
            
            processed_subset[file] = [signal, length]
  
            # Calculate and print progress
            percent = np.round(100*(i+1)/n, 2)
            print(f'{percent}%', end='\r')
            
        except Exception as exc:
            print(f"Error processing file: {futures[future]}, Error: {exc}")
            
    print(f"[{j}/{m}]")
    processed_subsets.append(dict(sorted(processed_subset.items())))
  
  return processed_subsets
  
  
  
  
  

def generate_city_noise(df, wham_folder, split, duration):
  
  """
  Generates the city noise signal by concatenating audio files until the desired 
  duration is reached.

  Inputs:
    - df          : The dataframe containing the metadata
    - wham_folder : The path to the WHAM noise folder
    - split       : The dataset split (train/val/test)
    - duration    : The desired duration of the city noise in seconds

  Saves the generated city noise in the appropriate directory.
  """
  
  processed_subsets = process_subsets(df, wham_folder)
  subsets = [list(dict.values()) for dict in processed_subsets]
  m = len(processed_subsets)
  len_sum = 0
  i = 0
  j = 0
  city_noise = []
  while len_sum < duration*16000:
    
    signal = subsets[i][j][0]
    len_sum += subsets[i][j][1]
    i += 1
    
    if i == m:
      j += 1
      i %= m

    city_noise.append(signal)
  
  city_noise_flat = np.vstack(city_noise)
  city_noise_flat = city_noise_flat[:duration*16000]
    
  save_dir = os.path.join("LibriVAD","Files","Noises","City_noise","City_noise_" + split + ".wav")
  
  wav.write(save_dir, 16000, city_noise_flat)





  
if __name__ == "__main__":

  wham_folder = sys.argv[1]
  metadata = os.path.join(wham_folder, "metadata")
  noise_meta_tr = os.path.join(metadata, "noise_meta_tr.csv")
  noise_meta_cv = os.path.join(metadata, "noise_meta_cv.csv")
  noise_meta_tt = os.path.join(metadata, "noise_meta_tt.csv")
  
  df_tr = pd.read_csv(noise_meta_tr)
  df_cv = pd.read_csv(noise_meta_cv)
  df_tt = pd.read_csv(noise_meta_tt)
  
  
  generate_city_noise(df_cv, os.path.join(wham_folder, "cv"), "val", 1800)
  generate_city_noise(df_tt, os.path.join(wham_folder, "tt"), "test", 1800)
  generate_city_noise(df_tr, os.path.join(wham_folder, "tr"), "train", int(3*3600))

  # Converting from stereo to mono
  for split in ["val", "test", "train"]:
    
    loc_dir = os.path.join("LibriVAD","Files","Noises","City_noise","City_noise_" + split + ".wav")

    sound = AudioSegment.from_wav(loc_dir)
    sound = sound.set_channels(1)
    sound.export(loc_dir, format="wav")
  
