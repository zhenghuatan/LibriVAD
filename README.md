# LibriVAD Dataset Generation

This repository contains the scripts necessary to generate the LibriVAD dataset, a large-scale, noise-augmented dataset for Voice Activity Detection (VAD) based on the LibriSpeech corpus.

## Setup LibriVAD Requirements

1.  **Install Python Libraries:** Navigate to the `LibriVAD` directory and install the required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Source Data:** Run the `setup.py` script and choose the LibriSpeech mirror closest to your location from `[EU, USA, CN]`. This will download the necessary splits of the LibriSpeech dataset, as well as the `Forced_alignments` and `Noises` data from Hugging Face.

    *EU Example:*
    ```bash
    python setup.py EU
    ```
    After downloading `train-clean-100`, `dev-clean`, and `test-clean`, the local setup should take a few minutes.

## Creating the LibriVAD Dataset

3.  **Generate the Dataset:** After the setup is complete, run the `create_LibriVAD.py` script. Choose the size of the generated dataset: `"small"`, `"medium"`, or `"large"`.

    The approximate sizes of the final dataset are **15GB**, **150GB**, and **1.5TB** respectively.

    *Small Dataset Example:*
    ```bash
    python create_LibriVAD.py small
    ```
    The generation process can take several hours, depending on the chosen size and your machine's performance.

> **Warning:** The initial space requirement for the setup process (before generating the final dataset) is approximately **61GB**. Please ensure you have sufficient disk space.

## Noisy Signals

The noise used for the generation of LibriVAD can be downloaded from https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/main/Files/Noises.zip

Dataset Hugging Face page: https://huggingface.co/datasets/LibriVAD/LibriVAD/

## Citation

I. Stylianou, A.K. Sarkar, N. Dawalatabad, J. Glass, and Z.-H. Tan, "LibriVAD: A Scalable Open Dataset with Deep Learning Benchmarks for Voice Activity Detection," arXiv preprint arXiv:2512.17281 (2025). [url](https://arxiv.org/abs/2512.17281)
