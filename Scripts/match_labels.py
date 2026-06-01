import glob
import os
from pathlib import Path

SNRs = [-5, 0, 5, 10, 15, 20]
Datasets = ["LibriSpeech", "LibriSpeechConcat"]
Noises = ["Babble_noise", "SSN_noise", "Domestic_noise", "Nature_noise", 
          "Office_noise", "Public_noise","Street_noise","Transport_noise", 
          "City_noise"]
Splits = ["train-clean-100", "dev-clean", "test-clean"]


def generate_match_txt(file, dataset, split):
  
  """
  Generates a 'label_match.txt' file for a given reference audio file.
  """
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent
  label_base_dir = project_root / "Files" / "Labels" / dataset / split
  
  file_path = Path(file)
  title = file_path.name
  label_filename = file_path.with_suffix(".npy").name

  # Reconstruct speaker and chapter folders from the filename.
  if "_+_" in title:
      # LibriSpeechConcat file: PART1_+_PART2.wav
      # These files are saved in the folder of PART2.
      name_parts = file_path.stem.split('_+_')
      part1_stem = name_parts[0]
      part2_info = name_parts[1]
      
      p1_parts = part1_stem.split('-')
      p2_parts = part2_info.split('-')
      
      if len(p2_parts) == 3: # Case 3: Different speaker
          speaker, chapter = p2_parts[0], p2_parts[1]
      elif len(p2_parts) == 2: # Case 2: Same speaker, different chapter
          speaker, chapter = p1_parts[0], p2_parts[0]
      else: # Case 1: Same speaker, same chapter
          speaker, chapter = p1_parts[0], p1_parts[1]
  else:
      # Standard LibriSpeech file: SPEAKER-CHAPTER-FILE.wav
      parts = title.split("-")
      speaker, chapter = parts[0], parts[1]

  label = label_base_dir / speaker / chapter / label_filename
  
  # The match file will be saved in the same directory as the input audio file.
  save_dir = file_path.parent
  match_file_path = save_dir / 'label_match.txt'
  with open(match_file_path, 'w') as f:
    f.write(str(file_path) + "\n\n" + str(label))


if __name__ == "__main__":
  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parent
   
  for dataset in Datasets:
    for split in Splits:
      for noise in Noises:
        for snr in SNRs:
          results_dir = project_root / 'Results' / dataset / split / noise / str(snr)

          # Use rglob to find all files in the directory
          files = list(results_dir.glob('*'))

          if not files:
              # print(f"Warning: Directory not found or empty, skipping: {results_dir}")
              continue

          files.sort()
          
          # Process each file to generate its own match file.
          for f in files:
              if f.suffix == ".wav":
                  generate_match_txt(str(f), dataset, split)

  print("Label matching files generated successfully.")
