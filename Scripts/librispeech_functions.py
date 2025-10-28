import textgrids
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

def scale_signal(signal):
    """
    Scales a signal to the 16-bit integer range [-32767, 32767].
    """
    # Avoid division by zero if the signal is pure silence
    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal.astype(np.int16)
    
    return np.int16(signal / max_abs * 32767)


def remove_unaligned_paths(paths, unaligned_txt):
    """
    Removes paths from a list that are listed in the unaligned.txt file.
    """
    # Read the list of unaligned file stems
    with open(unaligned_txt, 'r') as f:
        unaligned_stems = {line.strip().split(' ')[0] for line in f}

    # Filter the original path list, keeping only paths whose stems are NOT in the unaligned set
    aligned_paths = [
        path for path in paths if Path(path).stem not in unaligned_stems
    ]
    
    return aligned_paths


def find_alignment(sgn_path_str, alignments_dir_str):
    """
    Finds the corresponding .TextGrid alignment file for a given audio file path.

    Returns:
        Path: The path to the .TextGrid file.
    """
    sgn_path = Path(sgn_path_str)
    alignments_dir = Path(alignments_dir_str)

    # The structure is typically alignments_dir / speaker_id / chapter_id / file_stem.TextGrid
    speaker_id = sgn_path.parts[-3]
    chapter_id = sgn_path.parts[-2]
    
    # Create the final path, replacing the extension with .TextGrid
    alignment_path = alignments_dir / speaker_id / chapter_id / sgn_path.with_suffix(".TextGrid").name
    
    return alignment_path


def find_timeframes(wav_path_str, alignments_dir_str):
    """
    Finds forced alignment words and their end times from a .TextGrid file.

    Returns:
        tuple: (A numpy array of end times, a list of corresponding words).
    """
    # Find the alignment file path
    alignment_path = find_alignment(wav_path_str, alignments_dir_str)

    # Check if the alignment file exists before trying to parse it
    if not alignment_path.is_file():
        # print(f"Warning: Alignment file not found: {alignment_path}")
        return np.array([]), []

    # Parse the .TextGrid file
    grid = textgrids.TextGrid(alignment_path)
    words_tier = grid['words']


    # Extract the end times and corresponding text for each word/segment
    timeframes = np.array([interval.xmax for interval in words_tier], dtype=np.float32)
    words_txt = [interval.text for interval in words_tier]

    return timeframes, words_txt


def extract_silence(wav_path, alignments_dir):
    """
    Extracts silent segments from a .wav file using its forced alignment.
    Args:
        wav_path (str or Path): Path to the .wav audio file.
        alignments_dir (str or Path): Directory containing the .TextGrid alignment files.
    Returns:
        np.array: A numpy array containing the concatenated silent audio.
    """
    sr, signal = wav.read(wav_path)

    # Get the time and word information from the alignment file
    timeframes, txt = find_timeframes(wav_path, alignments_dir)
    #print(f"Extracting silence from {wav_path}, found {len(txt)} segments.")

    # Find the indices of segments marked as silent ('')
    silence_indices = np.where(np.array(txt) == '')[0]

    # If no silent segments are found in the alignment file, return an empty array.
    if silence_indices.size == 0:
        return np.array([], dtype=signal.dtype)

    silence_intervals = []
    # Handle silence at the very beginning of the audio
    if silence_indices[0] == 0:
        start_time = 0.0
        end_time = timeframes[0]
        silence_intervals.append([start_time, end_time])
        # Remove the first index so we don't process it again in the loop
        silence_indices = silence_indices[1:]
    
    # For other silent segments, the silence is between the end of the previous word
    # and the end of the current silent segment.
    for i in silence_indices:
        # The silence starts at the end time of the previous segment (i-1)
        start_time = timeframes[i-1]
        end_time = timeframes[i]
        silence_intervals.append([start_time, end_time])

    # Convert time intervals (in seconds) to sample indices
    silence_intervals_samples = (np.array(silence_intervals) * sr).astype(np.int32)

    # Slice the corresponding silent segments from the audio signal and concatenate them
    silent_segments = [signal[start:end] for start, end in silence_intervals_samples]
    
    if not silent_segments:
        return np.array([], dtype=signal.dtype)

    return np.concatenate(silent_segments)


def remove_silence(signal, wav_path, label_path=None):
    """
    Removes silent parts of a signal using a pre-computed VAD label file.

    Args:
        signal (np.array): The audio signal from which to remove silence.
        wav_path (str or Path): Path to the .wav audio file (used to infer label path if not provided).
        label_path (str or Path, optional): Path to the .npy label file. If None, it will be inferred from wav_path."""
    if label_path is None:
        # Construct the label path from the wav path
        wav_p = Path(wav_path)
        parts = list(wav_p.parts)
        try:
            # Find the dataset name (e.g., 'LibriSpeech') and replace it with 'Labels'
            idx = parts.index(wav_p.parts[-5]) # e.g., 'train-clean-100'
            parts[idx-1] = "Labels"
        except (ValueError, IndexError):
            # Fallback for unexpected path structures
            parts[-6] = "Labels"
        
        parts[-1] = wav_p.with_suffix(".npy").name
        label_path = Path(*parts)
    
    label = np.load(label_path)
    
    # Keep only the parts of the signal where the label is 1 (speech)
    silence_omitted_signal = signal[np.where(label == 1)[0]]

    return silence_omitted_signal


def RMS(signal):
    """
    Computes the Root Mean Square of a signal waveform.
    """
    # Use 64-bit float for squaring to prevent overflow with int16 signals
    squared_signal = np.array(signal, dtype=np.float64)**2
    mean_squared_signal = np.mean(squared_signal)
    rms = np.sqrt(mean_squared_signal)

    return rms