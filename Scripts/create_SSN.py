
# Generates Speech-Shaped Noise (SSN) based on selected speakers from the
# libriligh-small. We do not provide the data here, but you can download it
# from https://github.com/facebookresearch/libri-light

# The librosa library is used for LPC analysis, but is not provided in the
# requirements.txt as it is only needed for this script.


import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import librosa
import glob
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the system path to import local modules.
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf


# --- Configuration ---
# The choice of speakers is based on Signal-to-Noise Ratio (SNR), Silence Ratio (SR), speech duration and the Genre of the audio source 
# (i.e. SNR > 15, SR < 0.1, speech duration > 10 min, Genre = Literature) 

# Dictionary mapping each data split to the specific speaker IDs used for the SSN.
SPEAKER_IDS = {
    "train": [
        "1085", "1401", "147", "152", "16", "1614", "1649", "2090", "2294", 
        "2333", "2368", "2769", "28", "3483", "3885", "4090", "4297", "4792"
    ],
    "val": ["4881", "66"],
    "test": ["684", "817"]
}

# The target duration in seconds for the final SSN signal of each split.
SEGMENT_DURATIONS_SEC = {
    "train": 3600,  # 60 minutes
    "val": 600,     # 10 minutes
    "test": 600     # 10 minutes
}

# The order of the LPC analysis.
LPC_ORDER = 12
# The sample rate for all audio processing.
SAMPLE_RATE = 16000

# Define the source and destination directories relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Files" / "Datasets" / "LibriSpeech"
OUTPUT_DIR = PROJECT_ROOT / "Files" / "Noises" / "SSN_noise"


def generate_ssn_for_split(split_name, speaker_ids, segment_duration_sec ):
    """
    Performs the full SSN generation pipeline for a single data split.

    Args:
        split_name (str): The name of the split (e.g., 'train', 'val', 'test').
        speaker_ids (list[str]): A list of speaker IDs to use for this split.
        segment_duration_sec  (int): The duration of the noise segment for each speaker.
    """
    print(f"\n--- Generating SSN for '{split_name}' split ---")

    all_speaker_ssn_segments = []

    # Iterate through each specified speaker to create their unique noise segment.
    for speaker_id in tqdm(speaker_ids, desc=f"Processing speakers for {split_name}"):
        
        # 1. Find all audio files for the current speaker.
        search_pattern = str(DATASET_DIR / '**' / speaker_id / '**' / '*.wav')
        speaker_files = glob.glob(search_pattern, recursive=True)

        if not speaker_files:
            print(f"\nWarning: No audio files found for speaker {speaker_id}. Skipping.")
            continue

        # 2. Load and concatenate all speech from this one speaker.
        speech_segments = []
        for path in speaker_files:
            try:
                sr, sgn = wav.read(path)
                if sr == SAMPLE_RATE:
                    speech_segments.append(sgn)
            except Exception as e:
                print(f"Could not read {path}: {e}")
        
        if not speech_segments:
            print(f"\nWarning: Could not load any valid audio for speaker {speaker_id}. Skipping.")
            continue
            
        speaker_full_signal = np.concatenate(speech_segments)

        # 3. Perform personalized LPC analysis for this speaker.
        lpc_coefficients = librosa.lpc(speaker_full_signal.astype(float), order=LPC_ORDER)

        # 4. Generate white noise for this segment.
        noise_duration = segment_duration_sec + 5 # Extra for edge effects
        white_noise = np.random.normal(0, 1, int(noise_duration * SAMPLE_RATE))

        # 5. Filter the noise with the speaker's personal LPC filter.
        speaker_ssn = signal.lfilter([1], lpc_coefficients, white_noise)
        
        # 6. Trim to the exact segment duration and append to the list.
        target_samples = int(segment_duration_sec * SAMPLE_RATE)
        all_speaker_ssn_segments.append(speaker_ssn[:target_samples])

    if not all_speaker_ssn_segments:
        print(f"FATAL: No SSN segments could be generated for the '{split_name}' split.")
        return

    # 7. Concatenate all individual speaker segments into the final composite signal.
    print("Concatenating all speaker segments...")
    final_ssn_composite = np.concatenate(all_speaker_ssn_segments)
    
    # 8. Normalize, scale, and save the final SSN file.
    print("Normalizing and saving the final SSN file...")
    final_ssn_scaled = lf.scale_signal(final_ssn_composite)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"SSN_noise_temp_{split_name}.wav"
    
    wav.write(output_path, SAMPLE_RATE, final_ssn_scaled)
    print(f"Successfully saved composite SSN to: {output_path}")

if __name__ == "__main__":
    # Check if the source LibriSpeech directory exists.
    if not DATASET_DIR.is_dir():
        print(f"FATAL: LibriSpeech directory not found at '{DATASET_DIR}'")
        print("Please ensure the dataset has been downloaded and converted to .wav.")
        sys.exit(1)

    # Loop through each split defined in the configuration and generate its SSN.
    for split in ["train", "val", "test"]:
        generate_ssn_for_split(
            split_name=split,
            speaker_ids=SPEAKER_IDS[split],
            segment_duration_sec=SEGMENT_DURATIONS_SEC[split]
        )
    
    print("\nAll SSN files have been generated successfully.")