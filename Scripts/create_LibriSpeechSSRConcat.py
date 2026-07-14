# Script used to create the LibriSpeechConcat dataset and its corresponding 
# true VAD Labels simultaneously, pairing and merging audio files from the 
# LibriSpeech dataset to precisely hit target Silence-to-Speech Ratios (SSR).

import glob
import scipy.io.wavfile as wav
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
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
    """
    if target_samples <= 0:
        return np.array([], dtype=np.int16)
        
    label = np.ones(target_samples, dtype=np.int16)

    if timeframes.size == 0:
        return np.zeros(target_samples, dtype=np.int16)

    is_silent = [w == '' or w is None for w in words]
    silent_indices = np.where(is_silent)[0]

    if silent_indices.size > 0:
        if silent_indices[0] == 0:
            end_sample = min(int(timeframes[0] * sample_rate), target_samples)
            label[0:end_sample] = 0
        
        for j in silent_indices:
            if j > 0: 
                start_sample = min(int(timeframes[j-1] * sample_rate), target_samples)
                end_sample = min(int(timeframes[j] * sample_rate), target_samples)
                label[start_sample:end_sample] = 0
    
    # Handle trailing silence.
    last_word_end_sample = int(timeframes[-1] * sample_rate)
    if last_word_end_sample < target_samples:
        label[last_word_end_sample:] = 0
    
    return label


def get_audio_and_silence_length(path_str, alignments_dir):
    sr, wav_data = wav.read(path_str)
    silence_data = lf.extract_silence(path_str, alignments_dir)
    sil_len = len(silence_data) if silence_data is not None else 0
    return len(wav_data), sil_len


def precalculate_lengths(list_of_paths, alignments_dir):
    path_info = {}
    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(get_audio_and_silence_length, p, alignments_dir): p 
            for p in list_of_paths
        }
        for future in tqdm(as_completed(future_to_path), total=len(list_of_paths), desc="Analyzing audio/silence lengths"):
            path = future_to_path[future]
            try:
                path_info[path] = future.result()
            except Exception as exc:
                print(f"Error analyzing lengths for {path}: {exc}")
                path_info[path] = (0, 0)
    return path_info


def generate_silence_signal(list_of_paths, alignments_dir):
    all_silence_segments = []

    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(lf.extract_silence, path, alignments_dir): path
            for path in list_of_paths
        }
        for future in tqdm(as_completed(future_to_path), total=len(list_of_paths), desc="Extracting silence"):
            try:
                silence_segment = future.result()
                if silence_segment is not None and len(silence_segment) > 0:
                    all_silence_segments.append(silence_segment)
            except Exception as exc:
                print(f"Error extracting silence: {exc}")
    
    if not all_silence_segments:
        return np.array([], dtype=np.int16)
    return np.concatenate(all_silence_segments)


def process_pair(task_info, full_silence, ssr, align_split_dir):
    """
    Processes a pair of audio files, concatenates them to reach the desired SSR,
    and generates the exact label for the concatenated file concurrently.
    """
    path1_str, path2_str, silence_start, inbetween_silence_len = task_info
    
    path1 = Path(path1_str)
    path2 = Path(path2_str)

    sr, wav1 = wav.read(path1)
    sr, wav2 = wav.read(path2)
    n_silence = len(full_silence)

    if inbetween_silence_len > n_silence:
        raise ValueError('Generated silence signal is too short for the required length.')

    # 1. AUDIO CONCATENATION
    if inbetween_silence_len > 0:
        silence_end = silence_start + inbetween_silence_len
        if silence_end > n_silence:
            first_chunk = full_silence[silence_start:]
            second_chunk = full_silence[:(silence_end - n_silence)]
            inbetween_silence = np.concatenate((first_chunk, second_chunk))
        else:
            inbetween_silence = full_silence[silence_start:silence_end]
        
        sample_concat = np.concatenate((wav1, 0.001 * inbetween_silence, wav2))
    else:
        sample_concat = np.concatenate((wav1, wav2))
        
    output = lf.scale_signal(sample_concat)
    
    # 2. LABEL GENERATION
    timeframes1, words1 = lf.find_timeframes(str(path1), str(align_split_dir))
    label1 = _generate_label_from_timeframes(timeframes1, words1, len(wav1), sample_rate=sr)
    
    timeframes2, words2 = lf.find_timeframes(str(path2), str(align_split_dir))
    label2 = _generate_label_from_timeframes(timeframes2, words2, len(wav2), sample_rate=sr)
    
    if inbetween_silence_len > 0:
        inbetween_silence_label = np.zeros(inbetween_silence_len, dtype=np.int16)
        label_concat = np.concatenate((label1, inbetween_silence_label, label2))
    else:
        label_concat = np.concatenate((label1, label2))
    
    # 3. CONSTRUCT OUTPUT PATHS FOR AUDIO AND LABEL
    case_type = "same_chapter"
    if path1.parts[-3] != path2.parts[-3]: 
        file_name = f"{path1.stem}_+_{path2.name}"
        case_type = "diff_speaker"
    elif path1.parts[-2] != path2.parts[-2]: 
        file_name = f"{path1.stem}_+_{'-'.join(path2.stem.split('-')[1:])}.wav"
        case_type = "diff_chapter"
    else: 
        file_name = f"{path1.stem}_+_{path2.stem.split('-')[-1]}.wav"

    parts2 = list(path2.parts)
    try:
        idx_dataset = parts2.index("Datasets")
        idx_libri = parts2.index("LibriSpeech")
        
        # Audio Path
        parts_wav = parts2.copy()
        parts_wav[idx_libri] = f"LibriSpeechConcat_SSR_{ssr}"
        parts_wav[-1] = file_name
        final_wav_path = Path(*parts_wav)
        
        # Label Path
        parts_lbl = parts_wav.copy()
        parts_lbl[idx_dataset] = "Labels"
        final_label_path = Path(*parts_lbl).with_suffix(".npy")
        
    except ValueError:
        print(f"Warning: Base path structures not found for {path2}")
        return None, None, None

    # Save Audio
    final_wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav.write(final_wav_path, sr, output)
    
    # Save Label
    final_label_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(final_label_path, label_concat)
    
    return case_type, str(final_wav_path), str(final_label_path)


def create_LibriSpeechConcat(list_of_paths, silence_path, ssr, path_info, align_split_dir):
    sr, full_silence = wav.read(silence_path)
    n_silence = len(full_silence)
    
    if len(list_of_paths) % 2 != 0:
        list_of_paths.pop()

    pairs = [(list_of_paths[i], list_of_paths[i+1]) for i in range(0, len(list_of_paths), 2)]
  
    tasks = []
    current_silence_start = 0
    
    ds_total_speech = 0
    ds_total_silence = 0
    generated_files = {"diff_speaker": [], "diff_chapter": [], "same_chapter": []}
    
    for path1, path2 in pairs:
        len1, sil_len1 = path_info[path1]
        len2, sil_len2 = path_info[path2]
        
        speech_len1 = max(0, len1 - sil_len1)
        speech_len2 = max(0, len2 - sil_len2)
        total_speech = speech_len1 + speech_len2
        total_existing_silence = sil_len1 + sil_len2
        
        target_total_silence = total_speech * ssr
        inbetween_silence_len = int(target_total_silence - total_existing_silence)
        if inbetween_silence_len < 0:
            inbetween_silence_len = 0
            
        tasks.append((path1, path2, current_silence_start, inbetween_silence_len))
        
        ds_total_speech += total_speech
        ds_total_silence += (total_existing_silence + inbetween_silence_len)
        
        if inbetween_silence_len > 0:
            current_silence_start = (current_silence_start + inbetween_silence_len) % n_silence

    with ThreadPoolExecutor() as executor:
        # Pass align_split_dir to process_pair for label generation
        future_to_task = {
            executor.submit(process_pair, task, full_silence, ssr, align_split_dir): task 
            for task in tasks
        }
    
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Creating Dataset & Labels (SSR={ssr})"):
            try:
                case_type, saved_wav, saved_lbl = future.result()
                if case_type:
                    generated_files[case_type].append((saved_wav, saved_lbl))
            except Exception as exc:
                print(f"Error processing pair: {exc}")
                
    return ds_total_speech, ds_total_silence, generated_files


def plot_waveform_and_labels(wav_path, label_path, ssr, case_name, output_dir):
    """
    Plots the true VAD label directly extracted from the newly generated .npy file 
    over the audio waveform to verify structural correctness.
    """
    sr, audio = wav.read(wav_path)
    label = np.load(label_path)
    
    # Normalize audio to float32 between -1 and 1
    audio_norm = audio.astype(np.float32) / np.max(np.abs(audio))
    time_axis = np.arange(len(audio)) / sr
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio_norm, label="Audio Signal", color="gray", alpha=0.8)
    # Plot the label
    plt.plot(time_axis, label * 0.8, label="True VAD Label", color="blue", alpha=0.7, linewidth=1.5)
    
    plt.title(f"True Label Verification - SSR: {ssr} | Case: {case_name}\nFile: {Path(wav_path).name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude / True VAD")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    plot_path = output_dir / f"True_Verification_SSR_{ssr}_{case_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Saved true label plot: {plot_path.name}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    wav_dir = project_root / "Files" / "Datasets" / "LibriSpeech"
    silence_path = wav_dir / "train100_silence.wav"
    alignments_dir = project_root / "Files" / "Forced_alignments" / "librispeech_alignments"
    unaligned_path = alignments_dir / "unaligned.txt"
    plots_dir = project_root / "Verification_Plots"
    plots_dir.mkdir(exist_ok=True)

    print("\nSearching LibriSpeech files...")
    paths_train = sorted(glob.glob(str(wav_dir / 'train-clean-100' / '**' / '*.wav'), recursive=True))
    paths_val = sorted(glob.glob(str(wav_dir / 'dev-clean' / '**' / '*.wav'), recursive=True))
    paths_test = sorted(glob.glob(str(wav_dir / 'test-clean' / '**' / '*.wav'), recursive=True))
    
    paths_train = lf.remove_unaligned_paths(paths_train, unaligned_path)

    print("\nGenerating silence signal...")
    silence_signal = generate_silence_signal(paths_train, alignments_dir / "train-clean-100")
    wav.write(silence_path, 16000, silence_signal)

    print("\nPre-calculating pure speech & silence amounts for train-clean-100...")
    train_path_info = precalculate_lengths(paths_train, alignments_dir / "train-clean-100")
    
    print("Pre-calculating pure speech & silence amounts for dev-clean...")
    val_path_info = precalculate_lengths(paths_val, alignments_dir / "dev-clean")
    
    print("Pre-calculating pure speech & silence amounts for test-clean...")
    test_path_info = precalculate_lengths(paths_test, alignments_dir / "test-clean")

    # Define the range of SSR values to explore
    ssr_values = [0.4, 0.6, 0.8, 1.0]

    for ssr in ssr_values:
        print(f"\n==============================================")
        print(f"Processing datasets & labels for SSR = {ssr}")
        print(f"==============================================")

        total_speech_dataset = 0
        total_silence_dataset = 0
        all_generated_files = {"diff_speaker": [], "diff_chapter": [], "same_chapter": []}

        # 1. Train
        sp1, sil1, files1 = create_LibriSpeechConcat(paths_train, silence_path, ssr, train_path_info, alignments_dir / "train-clean-100")
        total_speech_dataset += sp1
        total_silence_dataset += sil1
        for k in all_generated_files: all_generated_files[k].extend(files1[k])

        # 2. Dev
        sp2, sil2, files2 = create_LibriSpeechConcat(paths_val, silence_path, ssr, val_path_info, alignments_dir / "dev-clean")
        total_speech_dataset += sp2
        total_silence_dataset += sil2
        for k in all_generated_files: all_generated_files[k].extend(files2[k])

        # 3. Test
        sp3, sil3, files3 = create_LibriSpeechConcat(paths_test, silence_path, ssr, test_path_info, alignments_dir / "test-clean")
        total_speech_dataset += sp3
        total_silence_dataset += sil3
        for k in all_generated_files: all_generated_files[k].extend(files3[k])

        # --- Debugging & Verification ---
        print("\n--- Label Statistics & Verification ---")
        print(f"Target SSR            : {ssr}")
        measured_ssr = total_silence_dataset / total_speech_dataset if total_speech_dataset > 0 else 0
        print(f"Measured Overall SSR  : {measured_ssr:.4f}")
        
        print("\n--- Visual Verification (Plotting True Labels) ---")
        # Plot 1 Case 2 (Different Chapter, same speaker) if available
        if all_generated_files["diff_chapter"]:
            sample_wav, sample_lbl = random.choice(all_generated_files["diff_chapter"])
            plot_waveform_and_labels(sample_wav, sample_lbl, ssr, "Different_Chapter", plots_dir)
            
        # Plot 1 Case 3 (Different Speakers) if available
        if all_generated_files["diff_speaker"]:
            sample_wav, sample_lbl = random.choice(all_generated_files["diff_speaker"])
            plot_waveform_and_labels(sample_wav, sample_lbl, ssr, "Different_Speaker", plots_dir)

    print("\nAll LibriSpeechConcat datasets and true labels completed successfully.")