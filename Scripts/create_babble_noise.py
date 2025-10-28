
# This script was created in order to generate the babble noise signal
# for LibriVAD. Requires access to the LibriLight medium dataset 
# (321Gb-Unzipped/299Gb-Zipped). In order to avoid loading the files, the 
# ready-to-use babble noise is provided separately

import sys
import json
import librosa
import numpy as np
import glob
import scipy.io.wavfile as wav
import os 
from rVADfast import rVADfast

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf

vad = rVADfast()


def evaluate_file(LibriLight_file, snr_threshold, sr_threshold = .15, 
                  Verbose = False):

    """
    Function for evaluating if a LibriLight file meets the requirements to be 
    used for the babble noise creation
    
    Inputs: 
        - LibriLight_file : The path of the LibriLight .flac file (str)
        - snr_threshold   : The SNR threshold for filtering signals (int).
                            Indicates how clean the the signals should be
        - sr_threshold    : The Silence Ratio threshold for filtering signals (int).
                           Indicates what percentage of speech the the signals 
                           should contain
        - Verbose         : Boolean indicator of whether to print message
                            if criteria are met or not (Bool)

    Returns True or False depending if the file meets the criteria or not
    """

    # Opening the corresponding to the flac, json file 
    f = open(LibriLight_file[:-4]+"json")
    data = json.load(f)
    f.close()

    # Extracting information of interest
    snr = data["snr"]
    genre = data['book_meta']['meta_genre']
    speech_seg = np.array(data["voice_activity"])
    signal_duration = speech_seg[-1,1]

    # Calculating duration of speech and silence ratio
    speech_duration = np.sum(speech_seg[:,1] - speech_seg[:,0])
    silence_ratio = 1 - speech_duration/signal_duration

    # Evaluating whether file meets the requirements
    flag = 0
    if snr > snr_threshold and silence_ratio < sr_threshold:
        flag = 1

    if flag == 1:
        if Verbose:
            print(f"File {LibriLight_file} meets requirements for usage\n")

        return True

    else:
        if Verbose:
            print(f"File ({LibriLight_file}) doesn't meet requirements for usage\n")
            print(f"SNR: {snr}")
            print(f"Silence ratio: {silence_ratio}")
            print(f"Genre: {genre}\n")

        return False


def dynamic_range_compression(audio, threshold = -10, ratio = 4):

    """
    Basic dynamic range compression of audio function
    
    Inputs:
        - audio         : The input audio signal (np.array)
        - threshold     : The threshold level above which compression is applied 
                                    (float)
        - ratio         : The compression ratio (float)
    
    Output:
        - compressed_audio : The compressed audio signal
    """

    # Convert threshold from db to 16 bit sound file amplitude
    threshold_amp = 10**(threshold/20)*32767

    # Apply compression
    compressed_audio = np.copy(audio)
    above_threshold = audio > threshold_amp
    compressed_audio[above_threshold] = threshold_amp+(audio[above_threshold]-threshold_amp)/ratio

    bellow_threshold = audio < -threshold_amp
    compressed_audio[bellow_threshold] =-threshold_amp+(audio[bellow_threshold]+threshold_amp)/ratio

    # And scaling it up again
    output = lf.scale_signal(compressed_audio)

    return output



def extract_silence(path_to_audiofile, return_og = False):

    """
    Function for utilising the rVAD model for identifying the silent parts of 
    a clean audio file. Returns the waveform of the audio file, but with the 
    silent parts removed.

    Inputs:
        - path_to_audiofile : The path to the clean audio file (str)
        - return_og         : Indicator whether to return the original signal or
                              not (boolean)
    Outputs:
        - silence_omitted_sgn: A np.array waveform containing only the speech parts
                               of the original input signal (np.array)
        - og_sgn             : The original signal waveform (np.array)
        - sr                 : The sampling rate of the audio file (int)
    """

    # Reading the audio file and extracting VAD labels per 10ms 
    og_sgn, sr = librosa.load(path_to_audiofile, sr = None)
    vad_labels, _ = vad(og_sgn, sr)


    # Adjusting the VAD labels from every 10ms to each sample
    label_ps = np.zeros(len(og_sgn))
    samp_per_10ms = int(sr/100)
    for i in range(len(vad_labels)):
        if vad_labels[i] == 1:
            label_ps[samp_per_10ms*i:samp_per_10ms*(i+1)] = 1

    # Removing silent samples from the original waveform
    silence_omitted_sgn = og_sgn[label_ps.astype(bool)]


    if return_og:
        return silence_omitted_sgn, og_sgn, sr

    return silence_omitted_sgn, sr





def create_babble_noise(list_of_paths, duration, write = False,
                        save_dir = None, title = None, s_rate = 16000, 
                        remove_silence = True):
    """
    Function for creating babble noise. 
    
    The noise is constructed by mixing 6 channels of LibriLight signals. 
    
    Each channel is created by concatenating filtered speech signals. The 
    first 30 and final 10 seconds of appropriate signals are omitted, due to
    repetitiveness of the intro and outro of some LibriLight files.
    
    The babble noise signal is constructed so that the same speaker does not
    appear in more than one channels. 
    
    Energy standardization is applied to the signals in order to reduce the 
    discrepancy in loudness between them. This results in a better mixed babble 
    noise signal.
    
    Inputs: 
        - list_of_paths  : A list of LibrisLight file paths, based on which 
                           the babble noise will be created
        - duration       : The desired duration of the output noise file 
                           in seconds
        - write          : Boolean indicator of wether to save the babble
                           noise as a .wav file
        - save_dir       : The directory of the output .wav babble noise
        - title          : The title of the output file
        - s_rate         : Sampling rate of saved signal
        - remove_silence : Boolean indicator of wether to remove the silence of
                           the individual signals or not
    Outputs: 
        - i              : The index of the first signal of the speaker after 
                           the last speaker that appears in the babble noise.
                           e.g. if last speaker_id was 10 and 
                           sgn_next = list_of_paths[i], then sgn_next is the
                           first signal of speaker 11
        - Speaker_ids    : A list of the ids of the speakers that where 
                           utilised
    """

    Speaker_ids = []

    babble_length = s_rate*duration
    num_voices = 6
    voices = np.zeros((num_voices, babble_length))

    i = 0

    speaker_id = list_of_paths[0].split('/')[-3]
    Speaker_ids.append(speaker_id)

    # Iterating over each channel
    for j in range(num_voices):

        print(f"Channel {j+1}/{num_voices}")

        # Initializing
        channel_len = 0
        concat_voices = []

        # Storing signals until the desired duration is reached
        while channel_len < babble_length:
                
            current_file = list_of_paths[i]
            silence_ratio = .15

            # If silence is manually removed the silence ratio condition is not 
            # utilized
            if remove_silence:
                silence_ratio = 1

            # If the current file satisfies the desired conditions
            if evaluate_file(current_file, snr_threshold = 15, 
                                             sr_threshold = silence_ratio) == True:

                # Either load the file and remove the silence
                if remove_silence:
                    signal, sr = extract_silence(current_file)

                # Or just load the file
                else:
                    signal, sr = librosa.load(current_file, sr = 16000)

                # Standardize signal to unit rms
                signal /= lf.RMS(signal)

                # Remove intro and outro in appropriate cases
                if len(signal) > 40*sr:
                    signal = signal[30*sr: -10*sr]    

                # Increase counter
                channel_len += len(signal)
                i += 1

                loading_percent = np.round(100*channel_len/babble_length,2)
                print(f"{np.min([loading_percent, 100.00])}%", end = "\r")

                concat_voices.append(signal)
                
            else:
             
                i += 1


        # Concatenating the stored signals
        concat_voices_flat = np.hstack(concat_voices)
        
        # Trimming channel to the desired duration and standardizing
        concat_voices_flat = concat_voices_flat[:babble_length]
        concat_voices_flat /= lf.RMS(concat_voices_flat)
        
        # Storing channel
        voices[j,:] = concat_voices_flat

        # Increasing index of path choice until next channel starts with a new 
        # speaker
        speaker_id = current_file.split('/')[-3]
        Speaker_ids.append(speaker_id)
        
        next_speaker_id = list_of_paths[i].split('/')[-3]

        while next_speaker_id == speaker_id:
                
            i += 1
            next_speaker_id = list_of_paths[i].split('/')[-3]
            
        if j < num_voices - 1:
        
            Speaker_ids.append(next_speaker_id)

    energy = np.mean(voices*voices, axis = 1)
    max_eng = np.max(energy)
    
    
    # Averaging channels to create babble noise
    average = np.array(np.mean(voices, axis = 0))
    babble_noise = lf.scale_signal(average)
 
    if write:
        wav.write(save_dir + "/" + title, s_rate, babble_noise)
        
    return i, Speaker_ids



if __name__ == "__main__":

    LibriLight_medium_dir = sys.argv[1] # Directory of the LibriLight files
    save_dir = sys.argv[2] # Directory for saving the babble noise splits
    
    # Listing and sorting the LibriLight file paths
    print("Searching for files...")
    paths = glob.glob(LibriLight_medium_dir+"/*/*/*.flac")
    paths.sort()
    

    # Creating Babble_noise_train.wav
    print("\nFor the Babble_noise_train signal:\n")
    end_index, speakers = create_babble_noise(paths, int(3*3600), write = True, 
                                              save_dir = save_dir,
                                              title = "Babble_noise_train.wav",
                                              s_rate = 16000)
     
    for i in range(0,len(speakers)-1,2):
    
        print(f"Channel {int((i+2)/2)} starts with speaker {speakers[i]} and ends with speaker {speakers[i+1]}")

    end_index += 1
    
    paths_val = paths[end_index:]

    # Creating Babble_noise_train.wav
    print("\nFor the Babble_noise_val signal:\n")
    end_index, speakers = create_babble_noise(paths_val, 1800, write = True,
                                              save_dir = save_dir,
                                              title = "Babble_noise_val.wav",
                                              s_rate = 16000)
    
    for i in range(0,len(speakers)-1,2):
    
        print(f"Channel {int((i+2)/2)} starts with speaker {speakers[i]} and ends with speaker {speakers[i+1]}")

    end_index += 1
    
    paths_test = paths_val[end_index:]
    
    # Creating Babble_noise_train.wav
    print("\nFor the Babble_noise_test signal:\n")
    end_index, speakers = create_babble_noise(paths_test, 1800, write = True,
                                              save_dir = save_dir,
                                              title = "Babble_noise_test.wav",
                                              s_rate = 16000)
    
    for i in range(0,len(speakers)-1,2):
    
        print(f"Channel {int((i+2)/2)} starts with speaker {speakers[i]} and "\
f"ends with speaker {speakers[i+1]}")
    
    
    
    
    
    