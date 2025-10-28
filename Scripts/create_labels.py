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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# Add the parent 'Scripts' directory to the system path to import local modules.
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf


def _generate_label_from_timeframes(timeframes, words, sample_rate=16000):
    """
    Helper function to generate a VAD label array from alignment data.

    Args:
        timeframes (np.array): Array of segment end times in seconds.
        words (list): List of corresponding words or text for each segment.
        sample_rate (int): The sample rate of the audio.

    Returns:
        np.array: A binary (0 or 1) VAD label array of dtype np.int16.
    """
    # If there are no timeframes, no label can be generated.
    if timeframes.size == 0:
        return np.array([], dtype=np.int16)
        
    # Initialize a label array of all ones (speech).
    total_samples = int(timeframes[-1] * sample_rate)
    label = np.ones(total_samples, dtype=np.int16)

    # Find segments that are marked as silent. The textgrids library may return
    # None for empty text fields, so we check for both possibilities.
    is_silent = [w == '' or w is None for w in words]
    silent_indices = np.where(is_silent)[0]

    # If silent segments exist, "paint" zeros over the corresponding parts of the label.
    if silent_indices.size > 0:
        # Handle silence at the very beginning of the file.
        if silent_indices[0] == 0:
            end_sample = int(timeframes[0] * sample_rate)
            label[0:end_sample] = 0
        
        # Handle all other silent segments.
        for j in silent_indices:
            if j > 0: # The start time is the end time of the previous segment.
                start_sample = int(timeframes[j-1] * sample_rate)
                end_sample = int(timeframes[j] * sample_rate)
                label[start_sample:end_sample] = 0
    
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
        timeframes, words = lf.find_timeframes(str(wav_path), str(Path(alignments_root_dir) / wav_path.parts[-4]))

        label = _generate_label_from_timeframes(timeframes, words)
        
    else:
        # --- LibriSpeechConcat file ---
        # The filename itself contains the names of the two source files separated by '_+_'.
        # e.g., "PART1_+_PART2.wav"
        concat_parts = wav_path.stem.split('_+_')
        part1_stem = concat_parts[0]
        part2_info = concat_parts[1]

        # Reconstruct the path to the first source file.
        # It is always in the same directory as the concatenated file.
        file1 = wav_path.parent / f"{part1_stem}.wav"

        # Reconstruct the path to the second source file.
        # The logic depends on the structure of the second part of the filename.
        part2_sub_parts = part2_info.split('-')
        len_parts2 = len(part2_sub_parts)
        
        # Grab the speaker ID from the current path (assumes .../speaker/chapter/file.wav)
        current_speaker = wav_path.parts[-3]

        if len_parts2 == 1:
             # Case 1: Same speaker, same chapter.
             # e.g., 5694-64038-0003_+_0004.wav -> part2_info is "0004"
             current_chapter = wav_path.parts[-2]
             file2_stem = f"{current_speaker}-{current_chapter}-{part2_info}"
             file2 = wav_path.parent / f"{file2_stem}.wav"

        elif len_parts2 == 2:
             # Case 2: Same speaker, DIFFERENT chapter.
             # e.g., 5694-64038-0003_+_64039-0004.wav -> part2_info is "64039-0004"
             new_chapter = part2_sub_parts[0]
             # Go up one level from the current chapter to the speaker directory.
             speaker_dir = wav_path.parent.parent
             file2_stem = f"{current_speaker}-{part2_info}"
             file2 = speaker_dir / new_chapter / f"{file2_stem}.wav"

        else:
             # Case 3: Different speaker entirely.
             # e.g., 19-198-0049_+_23-2938-0000.wav -> part2_info is "23-2938-0001"
             new_speaker = part2_sub_parts[0]
             new_chapter = part2_sub_parts[1]
             # Go up to the split directory (e.g., train-clean-100).
             split_dir = wav_path.parent.parent.parent
             file2 = split_dir / new_speaker / new_chapter / f"{part2_info}.wav"

        # Generate labels for each of the two source files.
        timeframes1, words1 = lf.find_timeframes(str(file1), str(Path(alignments_root_dir) / wav_path.parts[-4]))
        label1 = _generate_label_from_timeframes(timeframes1, words1)
        
        timeframes2, words2 = lf.find_timeframes(str(file2), str(Path(alignments_root_dir) / wav_path.parts[-4]))
        label2 = _generate_label_from_timeframes(timeframes2, words2)
        
        # Create a silent segment (label 0) to place between the two labels.
        inbetween_silence = np.zeros(int((len(label1) + len(label2)) / 4), dtype=np.int16)
        
        # Combine the labels to form the final concatenated label.
        label = np.concatenate((label1, inbetween_silence, label2))

    # --- Save the generated label ---
    if label.size > 0:
        # Construct the output path by replacing the dataset name with 'Labels'.
        parts = list(wav_path.parts)
        try:
            # Find the dataset folder name (e.g., 'LibriSpeech') and replace it.
            # We look 5 levels up from the file to find the dataset root.
            idx = parts.index(wav_path.parts[-5]) 
            parts[idx-1] = "Labels"
        except (ValueError, IndexError):
            # Fallback for unexpected path structures.
            print(f"Warning: Could not determine save path correctly for {wav_path}")
            return

        # Create the final .npy file path.
        label_path = Path(*parts).with_suffix(".npy")
        
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