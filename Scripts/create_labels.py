"""
Generates Voice Activity Detection (VAD) labels for the LibriSpeech and
LibriSpeechConcat datasets.

This script processes .wav files from both datasets, reads their corresponding
forced alignment .TextGrid files, and creates a binary label array for each.
The label is a frame-level annotation where '1' indicates speech and '0'
indicates silence.

The generated labels are saved as .npy files in a parallel 'Labels' directory,
preserving the original dataset structure. The process is parallelized using
a thread pool for efficiency.
"""
import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wav
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# Add the parent 'Scripts' directory to the system path to import local modules.
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf


def _generate_label_from_timeframes(timeframes, words, target_samples, sample_rate=16000):
    """
    Helper function to generate a VAD label array from alignment data.

    Args:
        timeframes (np.array): Array of segment end times in seconds.
        words (list): List of corresponding words or text for each segment.
        target_samples (int): The expected number of samples in the label.
        sample_rate (int): The sample rate of the audio.

    Returns:
        np.array: A binary (0 or 1) VAD label array of dtype np.int16.
    """
    # If there are no target samples, return empty.
    if target_samples <= 0:
        return np.array([], dtype=np.int16)
        
    # Initialize a label array of all ones (speech).
    label = np.ones(target_samples, dtype=np.int16)

    if timeframes.size == 0:
        # If no alignment is found, we assume the whole segment is silent.
        return np.zeros(target_samples, dtype=np.int16)

    # Find segments that are marked as silent.
    is_silent = [w == '' or w is None for w in words]
    silent_indices = np.where(is_silent)[0]

    # If silent segments exist, "paint" zeros over the corresponding parts of the label.
    if silent_indices.size > 0:
        # Handle silence at the very beginning of the file.
        if silent_indices[0] == 0:
            end_sample = min(int(timeframes[0] * sample_rate), target_samples)
            label[0:end_sample] = 0
        
        # Handle all other silent segments.
        for j in silent_indices:
            if j > 0: # The start time is the end time of the previous segment.
                start_sample = min(int(timeframes[j-1] * sample_rate), target_samples)
                end_sample = min(int(timeframes[j] * sample_rate), target_samples)
                label[start_sample:end_sample] = 0
    
    # IMPORTANT: Handle trailing silence.
    # Alignments often end before the actual audio file does.
    last_word_end_sample = int(timeframes[-1] * sample_rate)
    if last_word_end_sample < target_samples:
        label[last_word_end_sample:] = 0
    
    return label


def create_label_for_file(wav_path_str, alignments_root_dir):
    """
    Generates and saves a VAD label for a single .wav file, correctly
    handling both LibriSpeech and LibriSpeechConcat file types.

    Args:
        wav_path_str (str): The path to the source .wav signal.
        alignments_root_dir (str): The root directory of the LibriSpeech alignments.
    """
    wav_path = Path(wav_path_str)
    label = np.array([], dtype=np.int16)

    # Determine which dataset the file belongs to by checking the path.
    is_concat = "LibriSpeechConcat" in wav_path.parts
    
    if not is_concat:
        # --- Standard LibriSpeech file ---
        sr, audio = wav.read(wav_path)
        timeframes, words = lf.find_timeframes(str(wav_path), str(Path(alignments_root_dir) / wav_path.parts[-4]))

        label = _generate_label_from_timeframes(timeframes, words, len(audio))
        
    else:
        # --- LibriSpeechConcat file ---
        # The filename itself contains the names of the two source files separated by '_+_'.
        # e.g., "PART1_+_PART2.wav"
        concat_parts = wav_path.stem.split('_+_')
        part1_stem = concat_parts[0]
        part2_info = concat_parts[1]

        # Reconstruct the path to the first source file by parsing its stem.
        p1_parts = part1_stem.split('-')
        p1_speaker = p1_parts[0]
        p1_chapter = p1_parts[1]
        
        # Determine the split directory (e.g., dev-clean) from the output path.
        split_dir_name = wav_path.parts[-4]
        project_root = wav_path.parents[4]
        librispeech_root = project_root / "LibriSpeech" / split_dir_name
        
        file1 = librispeech_root / p1_speaker / p1_chapter / f"{part1_stem}.wav"

        # Reconstruct the path to the second source file.
        part2_sub_parts = part2_info.split('-')
        len_parts2 = len(part2_sub_parts)
        current_speaker = wav_path.parts[-3]

        if len_parts2 == 1:
             # Case 1: Same speaker, same chapter.
             current_chapter = wav_path.parts[-2]
             file2_stem = f"{current_speaker}-{current_chapter}-{part2_info}"
             file2 = librispeech_root / current_speaker / current_chapter / f"{file2_stem}.wav"

        elif len_parts2 == 2:
             # Case 2: Same speaker, DIFFERENT chapter.
             new_chapter = part2_sub_parts[0]
             file2_stem = f"{current_speaker}-{part2_info}"
             file2 = librispeech_root / current_speaker / new_chapter / f"{file2_stem}.wav"

        else:
             # Case 3: Different speaker entirely.
             new_speaker = part2_sub_parts[0]
             new_chapter = part2_sub_parts[1]
             file2 = librispeech_root / new_speaker / new_chapter / f"{part2_info}.wav"

        # Read source files to determine their exact length for duration matching.
        try:
            sr, wav1 = wav.read(file1)
            sr, wav2 = wav.read(file2)
        except Exception as e:
            print(f"Error reading source files for {wav_path.name}: {e}")
            return

        # Generate labels for each of the two source files, padded to their actual duration.
        align_split_dir = Path(alignments_root_dir) / split_dir_name
        timeframes1, words1 = lf.find_timeframes(str(file1), str(align_split_dir))
        label1 = _generate_label_from_timeframes(timeframes1, words1, len(wav1))
        
        timeframes2, words2 = lf.find_timeframes(str(file2), str(align_split_dir))
        label2 = _generate_label_from_timeframes(timeframes2, words2, len(wav2))
        
        # Create a silent segment (label 0) to place between the two labels.
        inbetween_silence_len = int((len(wav1) + len(wav2)) / 4)
        inbetween_silence = np.zeros(inbetween_silence_len, dtype=np.int16)
        
        # Combine the labels to form the final concatenated label.
        label = np.concatenate((label1, inbetween_silence, label2))

    # --- Save the generated label ---
    if label.size > 0:
        # Construct the output path by replacing 'Datasets' with 'Labels'.
        parts = list(wav_path.parts)
        try:
            idx = parts.index("Datasets")
            parts[idx] = "Labels"
        except ValueError:
            print(f"Warning: Could not determine save path correctly for {wav_path}")
            return

        # Create the final .npy file path.
        label_path = Path(*parts).with_suffix(".npy")
        
        # Ensure the destination directory exists.
        label_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(label_path, label)
        
        # Ensure the destination directory exists.
        label_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(label_path, label)
    

if __name__ == "__main__":
    # Define paths relative to this script's location for robustness.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Build all necessary base paths from the project root directory.
    datasets_root = project_root / "Files" / "Datasets"
    alignments_root_dir = project_root / "Files" / "Forced_alignments" / "librispeech_alignments"
    unaligned_path = alignments_root_dir / "unaligned.txt"
    
    datasets_to_process = ["LibriSpeech", "LibriSpeechConcat"]

    # Process each dataset sequentially.
    for i, dataset_name in enumerate(datasets_to_process):
        print(f"\nGenerating labels for {dataset_name} [{i+1}/{len(datasets_to_process)}]")
        
        dataset_path = datasets_root / dataset_name
        # Find all .wav files in the dataset directory recursively.
        wav_paths = sorted(glob.glob(str(dataset_path / '**' / '*.wav'), recursive=True))

        # For LibriSpeech, filter out files that don't have alignments.
        if dataset_name == "LibriSpeech":
            wav_paths = lf.remove_unaligned_paths(wav_paths, str(unaligned_path))
          
        if not wav_paths:
            print(f"No .wav files found for {dataset_name}. Skipping.")
            continue

        # Process all files for the current dataset in parallel.
        with ThreadPoolExecutor() as executor:
            future_to_path = {
                executor.submit(create_label_for_file, path, str(alignments_root_dir)): path
                for path in wav_paths
            }
            
            for future in tqdm(as_completed(future_to_path), total=len(wav_paths), desc=f"Processing {dataset_name}"):
                try:
                    future.result()
                except Exception as exc:
                    path = future_to_path[future]
                    print(f"Error processing file {Path(path).name}: {exc}")

    print("\nLabel generation completed.")