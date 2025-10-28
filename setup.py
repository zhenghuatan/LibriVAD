# Setup script for downloading and preparing the LibriSpeech dataset, 
# the LibriSpeechConcat dataset and their corresponding labels.

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm


# --- Configuration ---
# Dictionary mapping the region argument to its corresponding base URL.
URL_MIRRORS = {
    "US": "https://us.openslr.org/resources/12",
    "EU": "https://openslr.elda.org/resources/12",
    "CN": "https://openslr.magicdatatech.com/resources/12",
}

# List of files to download and extract.
FILES_TO_DOWNLOAD = [
    "train-clean-100.tar.gz",
    "dev-clean.tar.gz",
    "test-clean.tar.gz"
]

# --- Helper Function for Downloading ---
def download_file(url, destination):
    """Downloads a file from a URL to a destination with a progress bar."""
    print(f"Downloading {os.path.basename(destination)}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        sys.exit(1)


def run_command(command):
    """Runs a command and exits if it fails."""
    print(f"\n> Running script: {Path(command[-1]).stem}.py")
    try:
        # Using sys.executable ensures we use the python from the correct venv
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is Python in your PATH?")
        sys.exit(1)


def onerror(func, path, exc_info):
    """
    Source: https://stackoverflow.com/questions/2656322/shutil-rmtree-fails-on-windows-with-access-is-denied
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    
    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download and set up LibriSpeech dataset.")
    parser.add_argument(
        "region",
        choices=URL_MIRRORS.keys(),
        help="The download region (mirror) to use.",
    )
    args = parser.parse_args()

    # --- 2. Set up Paths and URLs ---
    base_url = URL_MIRRORS[args.region]
    
    # Use pathlib for cross-platform path handling
    base_dir = Path(__file__).parent.resolve()
    flac_dir = base_dir / "Files" / "Datasets" / "Flac"
    scripts_dir = base_dir / "Scripts"
    
    # Create the target directory if it doesn't exist
    flac_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using base URL: {base_url}")
    print(f"Downloading files to: {flac_dir}")

    # --- 3. Download, Extract, and Clean Up Archives ---
    for filename in FILES_TO_DOWNLOAD:
        file_url = f"{base_url}/{filename}"
        archive_path = flac_dir / filename

        # Download the file
        download_file(file_url, archive_path)

        # Extract the archive
        print(f"Extracting {filename}...")                                                       
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=flac_dir)
        
        # Remove the archive file after extraction
        print(f"Removing {filename}...")
        os.remove(archive_path)

    # --- 4. Run the Python Preprocessing Scripts ---
    run_command([sys.executable, str(scripts_dir / "create_LibriSpeech_wav.py")])
    
    # Clean up the Flac directory which is no longer needed
    print(f"Removing temporary directory: {flac_dir}")
    shutil.rmtree(flac_dir, onerror=onerror)
    
    run_command([sys.executable, str(scripts_dir / "create_LibriSpeechConcat.py")])
    run_command([sys.executable, str(scripts_dir / "create_labels.py")])

    print("\nSetup completed successfully!")


if __name__ == "__main__":
    main()