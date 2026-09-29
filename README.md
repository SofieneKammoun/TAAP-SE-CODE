# TAAP-SE: Test-Time Adaptation for Speech Enhancement

This repository provides the official implementation of the paper *[Test-time adaptation for speech enhancement with an autoregressive speech prior](https://arxiv.org/abs/2609.03622)*  authored by Sofiene Kammoun, Simon Leglaive, Xavier Alameda-Pineda and Timo Gerkmann, and published at IWAENC 2026.

[arXiv](https://arxiv.org/abs/2609.03622) | [Audio examples](https://sofienekammoun.github.io/TAAP-SE/) | [Bibtex](#citation)
___

The core idea is to first train a speech enhancement model to map noisy audio latents to clean audio latents in the continuous latent space of a pre-trained DAC [(Descript Audio Codec)](https://github.com/descriptinc/descript-audio-codec) model.
Separately, an autoregressive (AR) prior model is trained on a large corpus of clean speech to learn the distribution of clean speech latents. During inference on a new noisy utterance, the pre-trained NAR model is fine-tuned (adapted) using a loss function derived from the frozen AR prior model. This allows the enhancement model to adapt to the specific noise characteristics of the test sample without needing the corresponding clean reference audio.

### Key Components
*   **Representation Space:** Uses a pre-trained DAC model to encode audio into continuous embeddings.
*   **Non-Autoregressive Enhancement Model (`C_NAR_Model`):** A Conformer-based model that performs speech enhancement by mapping noisy embeddings to clean ones in a single forward pass.
*   **Autoregressive Prior Model (`C_AR_Model_Prior`):** A Conformer-based model that learns a full-covariance Gaussian distribution over sequences of clean speech embeddings.
*   **Test-Time Adaptation (TTA):** A procedure to fine-tune the enhancement model on a specific test sample. The adaptation is guided by minimizing the KL divergence between the enhancement model's output distribution and the learned prior's distribution.

## Models Architecture

The main models are located in the `Models/` directory.

*   **`Models/Conformer.py`**: A PyTorch implementation of the Conformer architecture, which combines multi-head self-attention with convolution to model both local and global dependencies. This serves as the backbone for both the prior and enhancement models.
*   **`Models/C_NAR.py`**: Defines the `C_NAR_Model`, a non-autoregressive Conformer-based network. It takes a sequence of noisy embeddings and directly outputs the predicted clean embeddings.
*   **`Models/C_AR_Prior.py`**: Defines the `C_AR_Model_Prior`, an autoregressive Conformer-based network. It models the next clean speech embedding in a sequence conditioned on the previous ones. It outputs the parameters (mean and log-variance) of a full-covariance Gaussian distribution, learning a powerful prior over clean speech.

## Scripts and Workflow

The repository is structured around three main executable scripts that represent the stages of the workflow.

### 1. Training the Speech Prior
**Script:** `Prior_Trainer.py`

This script trains the autoregressive prior model (`C_AR_Model_Prior`) on a dataset of clean speech (e.g., EARS). The model learns the statistical properties of clean speech embeddings from the DAC encoder.

-   **Input:** A directory of clean `.wav` files.
-   **Output:** A trained prior model checkpoint (`.pt` file).

### 2. Training the Enhancement Model
**Script:** `C_NAR_Trainer.py`

This script trains the non-autoregressive speech enhancement model (`C_NAR_Model`) on a dataset of parallel noisy and clean speech pairs (e.g., Libri2Mix). The model learns a mapping from noisy to clean embeddings using a Mean Squared Error (MSE) loss.

-   **Input:** Directories of paired noisy and clean `.wav` files.
-   **Output:** A trained speech enhancement model checkpoint (`.pt` file).

### 3. Test-Time Adaptation
**Script:** `Test_Time_Adapt.py`

This script performs test-time adaptation for a given noisy audio file. It loads the pre-trained enhancement model and the pre-trained prior model. For each new test file, it fine-tunes (adapts) the enhancement model by minimizing the KL divergence between its output and the distribution predicted by the frozen prior. This process specializes the model to the specific noise in the test file, improving enhancement quality.

-   **Input:** A directory of noisy `.wav` files to be enhanced, a pre-trained enhancement model checkpoint, and a pre-trained prior model checkpoint.
-   **Output:** Enhanced audio files saved to the `TTA_Audio/` directory.

## Setup and Usage

### Prerequisites
Install the required Python packages. It is recommended to use a virtual environment.
```bash
pip install torch numpy soundfile scipy librosa einops pystoi
pip install dac
```

### Configuration
Before running the scripts, you must configure the paths and hyperparameters within each file:

1.  **Download DAC Model:** Download the pre-trained 16kHz DAC model and update the `DAC_Model` variable in `C_NAR_Trainer.py`, `Prior_Trainer.py`, and `Test_Time_Adapt.py`.
2.  **Prepare Datasets:** Download the required datasets (e.g., Libri2Mix, EARS, or your custom data) and update the `DATA_PATHS` variables in the training scripts.
3.  **Checkpoints:** Ensure that checkpoint paths (`checkpoint_path`, `prior_path`) are correctly set, especially for `Test_Time_Adapt.py`, which requires pre-trained models.

### Running the Scripts
The scripts are designed for multi-GPU training using `torch.distributed`. They will automatically use all available CUDA devices.

1.  **Train the Prior Model:**
    ```bash
    python Prior_Trainer.py
    ```
    Checkpoints will be saved in the `Checkpoints/` directory.

2.  **Train the Enhancement Model:**
    ```bash
    python C_NAR_Trainer.py
    ```
    Checkpoints will be saved in the `Checkpoints/` directory. Audio samples from validation will be saved in a corresponding `*_Audio/` directory.

3.  **Perform Test-Time Adaptation:**
    Update `checkpoint_path` and `prior_path` in `Test_Time_Adapt.py` with the paths to the models trained in the previous steps.
    ```bash
    python Test_Time_Adapt.py
    ```
    The enhanced audio files will be saved in subdirectories within `TTA_Audio/`.

## File Structure

```
.
├── Models/
│   ├── C_AR_Prior.py      # Autoregressive speech prior model
│   ├── C_NAR.py           # Non-autoregressive enhancement model
│   └── Conformer.py       # Conformer block implementation
├── C_NAR_Trainer.py       # Script to train the C-NAR enhancement model
├── Prior_Trainer.py       # Script to train the C-AR prior model
├── Test_Time_Adapt.py     # Script for test-time adaptation and enhancement
└── Trainer.py             # Base trainer class with shared utilities
```
## Citation

If you use this code, please star the project and cite:
```
@inproceedings{kammoun2026test,
  title={Test-time adaptation for speech enhancement with an autoregressive speech prior},
  author={Kammoun, Sofiene and Leglaive, Simon and Alameda-Pineda, Xavier and Gerkmann, Timo},
  booktitle    = {19th International Workshop on Acoustic Signal Enhancement, {IWAENC} 2026},
  publisher    = {{IEEE}},
  year         = {2026}
}
```
