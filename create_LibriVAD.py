
# Authors: Ioannis Stylianou
# Date: September 2025

# Description: The main script for generating the LibriVAD dataset. 

import scipy.io.wavfile as wav
import glob 
import numpy as np
import os  
import sys 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf





def load_files(dataset, dataset_dir, split):
  
    """
    Function for loading the file paths of the corresponding LibriSpeech split
    
    Inputs:
    - dataset     : Indicates which dataset to use between LibriSpeech and 
                    LibriSpeechConcat (str)
    - dataset_dir : The directory of the dataset (str)
    - split       : Indicates which split to generate between "train", "val"
                    and "test" (str)
    """
    
    # Locating the paths of the utterances for each split
    if split == "train":   
        search_path = os.path.join(Path(dataset_dir), dataset, "train-clean-100", "*", "*", "*.wav")
        file_paths = glob.glob(search_path)

        
    elif split == "val":  
        search_path = os.path.join(Path(dataset_dir), dataset, "dev-clean", "*", "*", "*.wav")
        file_paths = glob.glob(search_path)
        
    elif split == "test":
        search_path = os.path.join(Path(dataset_dir), dataset, "test-clean", "*", "*", "*.wav")
        file_paths = glob.glob(search_path)        
    else:  
        raise ValueError('Split parameter should be a choice between "train", \
  "val" and "test"')

    # Removing (the few) utterances that are not paired with forced alignments
    if dataset == "LibriSpeech":
        alignments_dir = os.path.join("Files",
                                      "Forced_alignments")
        librispeech_unaligned = os.path.join(alignments_dir,
                                             "librispeech_alignments",
                                             "unaligned.txt")

        file_paths = lf.remove_unaligned_paths(file_paths, 
                                              librispeech_unaligned)   

    file_paths.sort()
    
    return file_paths






def extract_info(file):
  
    """
    Function for computing the speech Root Mean Square, the waveform, 
    and the length of a signal. These are required for accelerating the 
    dataset generation process
    """
    
    sr, signal = wav.read(file)
    
    # The RMS of the signal is computed only for the speech part of the 
    # utterance        
    omitted_silence_signal = lf.remove_silence(signal, file)
    rms = lf.RMS(omitted_silence_signal)

    return signal, rms, len(signal), file






def load_noise(noise, split):

    """
    Function for loading the noisy signals. Returns the waveform and the length
    of the noise
    """
    
    noise_path = os.path.join("Files","Noises", noise, noise+'_'+split+'.wav')
    
    sr, noise_sgn = wav.read(noise_path)
        
    return noise_sgn, len(noise_sgn)
  


def create_LibriVAD(dataset, dataset_dir, SNRs, Noises, split, size,
                    batch_size=500, save_dir=None):
    """
    Creates the LibriVAD dataset by mixing clean audio with various noises at
    specified SNRs. Samples individual files for different
    dataset sizes ('small', 'medium', 'large').

    Args:
      - dataset (str): 'LibriSpeech' or 'LibriSpeechConcat'.
      - dataset_dir (str): The root directory of the clean datasets.
      - SNRs (list): A list of Signal-to-Noise Ratios to generate.
      - Noises (list): A list of noises to mix in.
      - split (str): The data split to generate ('train', 'val', or 'test').
      - size (str): The target dataset size. 'small' (saves every 100th file),
                    'medium' (every 10th), or 'large' (every file).
      - batch_size (int): The number of files to process in memory at once.
      - save_dir (str, optional): The root directory to save the generated files.
                                  Defaults to "LibriVAD/Results" if None.
    """
    print(f"\nPreparing {split} files for {dataset} dataset (size: {size})...")
    
    # Setting up the output directory
    if save_dir is None:
        output_root = Path("Results")
    else:
        output_root = Path(save_dir)

    # Determine the sampling step based on size
    if size == "large":
        step = 1
    elif size == "medium":
        step = 10
    elif size == "small":
        step = 100
    else:
        raise ValueError('Size must be one of "small", "medium", or "large".')

    # Loading file paths
    file_paths_all = load_files(dataset, dataset_dir, split)
    file_paths = file_paths_all[::step]
    print(f"Total files to process after applying size filter: {len(file_paths)}")
    n_total_files = len(file_paths)
    sr = 16000
    
    # The main loop iterates through the data in batches for memory efficiency.
    for i_batch in range(0, n_total_files, batch_size):
        batch_paths = file_paths[i_batch : i_batch + batch_size]
        n_batch = len(batch_paths)
        
        print(f"\nProcessing batch {i_batch//batch_size + 1}/{(n_total_files - 1)//batch_size + 1}...")

        # Pre-fetch info for the current batch in parallel.
        Signals, RMSs, Lens, Files = [None] * n_batch, [None] * n_batch, [None] * n_batch, [None] * n_batch
        file_index_map = {path: i for i, path in enumerate(batch_paths)}
    
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_info, path): path for path in batch_paths}
            
            for future in tqdm(as_completed(futures), total=n_batch, desc="Loading batch data"):
                try:
                    signal, rms, sgn_len, file = future.result()
                    index = file_index_map[file]
                    Signals[index], RMSs[index], Lens[index], Files[index] = signal, rms, sgn_len, file
                except Exception as exc:
                    print(f"Error loading info for file: {futures[future]}, Error: {exc}")
          
        # Iterate through each noise and SNR combination.
        for noise in Noises:
            noise_sgn, noise_dur = load_noise(noise, split)
            noise_sgn_temp = np.tile(noise_sgn, 2) # For seamless recycling
            
            for snr in SNRs:
                print(f"\nMixing with {noise} at {snr} SNR...")
                noise_idx = 0
                
                for i in tqdm(range(n_batch), desc="Mixing audio"):

                    signal_length = Lens[i]
                    if noise_idx + signal_length > noise_dur:
                        noise_subset = noise_sgn_temp[noise_idx : noise_idx + signal_length]
                        noise_idx = (noise_idx + signal_length) % noise_dur
                    else:
                        noise_subset = noise_sgn[noise_idx : noise_idx + signal_length]
                        noise_idx += signal_length
                
                    rms = RMSs[i]
                    scale = rms / (lf.RMS(noise_subset) * (10**(snr / 20)))
                    signal = Signals[i]
                    corrupted_signal = signal + scale * noise_subset
                    output = lf.scale_signal(corrupted_signal)
                          
                    file_path = Path(Files[i])
                    
                    # Extract the split folder name (train, val, test)
                    split_folder = file_path.parts[-4]

                    save_dir_path = output_root / dataset / split_folder / noise / str(snr)
                    save_dir_path.mkdir(parents=True, exist_ok=True)
                    
                    save_path = save_dir_path / file_path.name
                    wav.write(save_path, sr, output)

    print("\Data generation completed!")

          
if __name__ == "__main__":
  
    import argparse

    parser = argparse.ArgumentParser(description="Generate the LibriVAD dataset.")
    parser.add_argument("size", choices=["small", "medium", "large"], help="Size of the dataset to generate.")
    parser.add_argument("--dataset_dir", type=str, default="Files/Datasets/", help="Root directory of datasets.")
    parser.add_argument("--snrs", nargs='+', type=int, default=[-5, 0, 5, 10, 15, 20], help="List of SNRs to use.")
    parser.add_argument("--batch_size", type=int, default=2900, help="Number of audio files to load into memory at once.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the generated dataset. If None, saves to LibriVAD/Results.")
    
    # To maintain the structure of the dataset these should be fixed; one can experiment with them however
    Datasets = ["LibriSpeech", "LibriSpeechConcat"]
    Noises = ["Babble_noise", "SSN_noise", "Domestic_noise", "Nature_noise", 
              "Office_noise", "Public_noise","Street_noise","Transport_noise", 
              "City_noise"]
    Splits = ["train", "val", "test"]
    
    args = parser.parse_args()

    
    for dataset in Datasets:
        print(f"\n--- Generating data based on the {dataset} dataset ---")
        for split in Splits:
            create_LibriVAD(dataset, 
                            args.dataset_dir, 
                            args.snrs, 
                            Noises, 
                            split, 
                            args.size, 
                            args.batch_size, 
                            args.save_dir)

  
  