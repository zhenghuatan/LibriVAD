# Script used to create the LibriSpeechConcat dataset by pairing and merging
# audio files from the LibriSpeech dataset.

import glob
import scipy.io.wavfile as wav
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# Add the parent 'Scripts' directory to the system path to import local modules.
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(scripts_path)
from Scripts import librispeech_functions as lf


def generate_silence_signal(list_of_paths, alignments_dir):
    """
    Generates a single, long silence signal by extracting and concatenating
    silent segments from multiple LibriSpeech audio files in parallel.

    Args:
        list_of_paths (list): Paths to audio files for silence extraction.
        alignments_dir (str): Directory containing forced alignment files.

    Returns:
        np.array: The resulting concatenated silence signal.
    """
    all_silence_segments = []

    # Use a thread pool to extract silence from files concurrently.
    with ThreadPoolExecutor() as executor:
        # Create a dictionary to map each future back to its file path for error reporting.
        future_to_path = {
            executor.submit(lf.extract_silence, path, alignments_dir): path
            for path in list_of_paths
        }

        # Use tqdm for a progress bar that updates as tasks complete.
        for future in tqdm(as_completed(future_to_path), total=len(list_of_paths), desc="Extracting silence"):
            try:
                silence_segment = future.result()
                if silence_segment is not None and len(silence_segment) > 0:
                    all_silence_segments.append(silence_segment)
            except Exception as exc:
                path = future_to_path[future]
                print(f"Error extracting silence from {path}: {exc}")
    
    # Combine all extracted segments into one continuous signal.
    if not all_silence_segments:
        print("Warning: No silence could be extracted. Returning empty signal.")
        return np.array([], dtype=np.int16)
        
    return np.concatenate(all_silence_segments)


def process_pair(task_info, full_silence):
    """
    Processes a pair of audio files, concatenates them with a slice of silence,
    and saves the resulting audio file. This function is designed to be run
    in a parallel thread and is fully self-contained.

    Args:
        task_info (tuple): Contains (path1_str, path2_str, silence_start_index).
        full_silence (np.array): The complete silence signal used for slicing.
    """
    path1_str, path2_str, silence_start = task_info
    
    # Use pathlib for robust, cross-platform path handling.
    path1 = Path(path1_str)
    path2 = Path(path2_str)

    sr, wav1 = wav.read(path1)
    sr, wav2 = wav.read(path2)

    # Calculate the required length of silence (25% of the combined audio length).
    n_silence = len(full_silence)
    inbetween_silence_len = int((len(wav1) + len(wav2)) / 4)

    if inbetween_silence_len > n_silence:
        raise ValueError('Generated silence signal is too short for the required length.')

    # Slice the silence signal, wrapping around to the beginning if necessary.
    silence_end = silence_start + inbetween_silence_len
    if silence_end > n_silence:
        first_chunk = full_silence[silence_start:]
        remaining_len = silence_end - n_silence
        second_chunk = full_silence[:remaining_len]
        inbetween_silence = np.concatenate((first_chunk, second_chunk))
    else:
        inbetween_silence = full_silence[silence_start:silence_end]
    
    # Create the new audio signal by concatenating wav1 -> silence -> wav2.
    sample_concat = np.concatenate((wav1, 0.001 * inbetween_silence, wav2))
    output = lf.scale_signal(sample_concat)
    
    # --- Robustly construct the output filename and path ---
    # The new filename is a combination of the two source filenames.
    if path1.parts[-3] != path2.parts[-3]: # Different speakers
        file_name = f"{path1.stem}_+_{path2.name}"
    elif path1.parts[-2] != path2.parts[-2]: # Different chapters, same speaker
        file_name = f"{path1.stem}_+_{'-'.join(path2.stem.split('-')[1:])}.wav"
    else: # Same chapter and speaker
        file_name = f"{path1.stem}_+_{path2.stem.split('-')[-1]}.wav"

    # Create the new path by modifying the path of the second file.
    parts2 = list(path2.parts)
    try:
        # Find and replace 'LibriSpeech' with 'LibriSpeechConcat'.
        idx = parts2.index("LibriSpeech")
        parts2[idx] = "LibriSpeechConcat"
    except ValueError:
        print("Warning: 'LibriSpeech' not found in path. Output path may be incorrect.")

    parts2[-1] = file_name
    final_path = Path(*parts2) # Reassemble the path from its parts.

    # Create the directory and save the new .wav file.
    final_path.parent.mkdir(parents=True, exist_ok=True)
    wav.write(final_path, sr, output)


def create_LibriSpeechConcat(list_of_paths, silence_path):
    """
    Creates the LibriSpeechConcat dataset for a given list of audio files.
    It pre-calculates all necessary information for each pair before starting
    the parallel processing, ensuring each task is independent.
    
    Args:
      - list_of_paths (list): A list of .wav file paths to be merged.
      - silence_path (Path): Path to the generated silence .wav file.
    """
    sr, full_silence = wav.read(silence_path)
    n_silence = len(full_silence)
    
    # Ensure the list of files is even by removing the last file if necessary.
    if len(list_of_paths) % 2 != 0:
        list_of_paths.pop()

    # Group the file paths into pairs.
    pairs = [(list_of_paths[i], list_of_paths[i+1]) for i in range(0, len(list_of_paths), 2)]
  
    # Pre-calculate the silence start index for each pair sequentially.
    # This makes each task independent and safe for parallel processing.
    tasks = []
    current_silence_start = 0
    for path1, path2 in pairs:
        # Read audio files to determine their length for the calculation.
        sr, wav1 = wav.read(path1)
        sr, wav2 = wav.read(path2)
        inbetween_silence_len = int((len(wav1) + len(wav2)) / 4)
        
        # Add the task info (paths and calculated start index) to the list.
        tasks.append((path1, path2, current_silence_start))
        
        # Update the next start index, wrapping around the silence buffer.
        current_silence_start = (current_silence_start + inbetween_silence_len) % n_silence

    # Process all the prepared tasks in parallel.
    with ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(process_pair, task, full_silence): task for task in tasks}
    
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Concatenating files"):
            try:
                future.result()
            except Exception as exc:
                task = future_to_task[future]
                path1_name = Path(task[0]).name
                path2_name = Path(task[1]).name
                print(f"Error processing pair ({path1_name}, {path2_name}): {exc}")


if __name__ == "__main__":
    # Define paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Build all necessary paths from the project root directory.
    wav_dir = project_root / "Files" / "Datasets" / "LibriSpeech"
    silence_path = wav_dir / "train100_silence.wav"
    alignments_dir = project_root / "Files" / "Forced_alignments" / "librispeech_alignments"
    unaligned_path = alignments_dir / "unaligned.txt"

    print("\nSearching LibriSpeech files...")
    # Find all .wav files in the specified subdirectories.
    paths_train = sorted(glob.glob(str(wav_dir / 'train-clean-100' / '**' / '*.wav'), recursive=True))
    paths_val = sorted(glob.glob(str(wav_dir / 'dev-clean' / '**' / '*.wav'), recursive=True))
    paths_test = sorted(glob.glob(str(wav_dir / 'test-clean' / '**' / '*.wav'), recursive=True))
    
    # Remove paths that do not have corresponding alignment files.
    paths_train = lf.remove_unaligned_paths(paths_train, unaligned_path)

    print("\nGenerating silence signal...")
    silence_signal = generate_silence_signal(paths_train, alignments_dir / "train-clean-100")
    wav.write(silence_path, 16000, silence_signal)

    # Process each data split sequentially.
    print("Preparing LibriSpeechConcat train-clean-100. This might take a few minutes...")
    create_LibriSpeechConcat(paths_train, silence_path)

    print("Preparing LibriSpeechConcat dev-clean...")
    #create_LibriSpeechConcat(paths_val, silence_path)

    print("Preparing LibriSpeechConcat test-clean...")
    create_LibriSpeechConcat(paths_test, silence_path)

    print("\nLibriSpeechConcat dataset creation completed.")