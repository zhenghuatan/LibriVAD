
import glob
import os

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
  
  label_base_dir = os.path.join("LibriVAD", "Files", "Labels", dataset, split)
  
  file = os.path.normpath(file)
  folders = file.split(os.sep)
  title = folders[-1]

  parts = title.split("-")
  
  label_filename = os.path.splitext(title)[0] + ".npy"
  label = os.path.join(label_base_dir, parts[0], parts[1], label_filename)
  
  # The match file will be saved in the same directory as the input audio file.
  save_dir = os.path.dirname(file)
  match_file_path = os.path.join(save_dir, 'label_match.txt')
  with open(match_file_path, 'w') as f:
    f.write(file+"\n\n"+label)


if __name__ == "__main__":
   
  for dataset in Datasets:
    for split in Splits:
      for noise in Noises:
        for snr in SNRs:
          results_dir = os.path.join('LibriVAD', 'Results', dataset, split, noise, str(snr))

          files = glob.glob(os.path.join(results_dir, '*'))

          if not files:
              print(f"Warning: Directory not found or empty, skipping: {results_dir}")
              continue

          files.sort()
          reference = files[-1]
          
          generate_match_txt(reference, dataset, split)

  print("Label matching files generated successfully.")
