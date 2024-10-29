# Autoencoder for Protein Structural Features

An autoencoder trained for dimensionality reduction of protein structural features. 
This was used for my research of *Common local protein structural motifs supported by specific amino acids.*
For more details, refer to my research [report](https://drive.google.com/uc?export=download&id=13sNe-aCBAS559GYmgL5GHN-8LVMoZ8H5) and the [SeqPredNN](https://doi.org/10.1186/s12859-023-05498-4) paper.

[<img height="350" style="display: block; margin-left: auto; margin-right: auto;" src="https://app.eraser.io/workspace/9celyqYK57U4tXAzgLt2/preview?elements=EEsNvKfrMNe9LUqS8-Urdg&amp;type=embed"/>](https://app.eraser.io/workspace/9celyqYK57U4tXAzgLt2?elements=EEsNvKfrMNe9LUqS8-Urdg)

## Requirements

- **Python**: 3.12
- **Packages**: `numpy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`, `scipy`, `biopython`, `umap-learn`

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate autoencoder
```

I recommend using [anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) to install the packages in a contained environment.

### 2. Feature Generation

```bash
python featurise.py -gm -o example_features examples/example_chain_list.csv examples/example_pdb_directory
```

This will generate features (using SeqPredNN's [featurise.py](https://github.com/falategan/SeqPredNN?tab=readme-ov-file#predicting-protein-sequences)) using PDB files from the example directory and create chain lists in the output directory.

### 3. Dimensionality Reduction

```bash
python inference.py example_features example_features/chain_list.txt pretrained_model/trained_model.pth -o inference --save_latent_space_vectors --plot_feature_space
```

This will load the pre-trained model using the specified features, run a forward pass through the autoencoder, and save the reconstructed features, latent vectors, and metrics.

### 4. Training a Model

```bash
python train_model.py example_features example_features/chain_list.txt -o model --balanced_sampling
```

This will train a new autoencoder model using the specified features, optionally customize the model architecture and training parameters, and save the trained model along with training metrics.

---

## Pre-trained Model Specifications

The pre-trained model was trained on 21,690 X-ray crystallographic protein chains (generated from the PISCES [server](https://dunbrack.fccc.edu/pisces/)) with resolution ≤ 2 Å, R-factor ≤ 0.25, chain lengths between 40-10,000 residues, and sequence identity < 90%. The model architecture consists of an input dimension of 180, hidden layers of 148 and 116 neurons, and a latent dimension of 84.

The `pretrained_model` directory includes:
- Trained model weights ([`trained_model.pth`](pretrained_model/trained_model.pth))
- PISCES dataset list ([`pisces_pdb_list.fasta`](pretrained_model/pisces_pdb_list.fasta), [`pisces_pdb_list`](pretrained_model/pisces_pdb_list))
- PDB chain list for feature extraction ([`chain_list_pdb.csv`](pretrained_model/chain_list_pdb.csv))
- Feature chain list for training ([`chain_list_training.txt`](pretrained_model/chain_list_training.txt))
- Feature chain list for testing ([`chain_list_testing.txt`](pretrained_model/chain_list_testing.txt))

## Examiner Details

The `examiner_details` directory includes:
- All scripts run on the HPC ([`scripts/`](examiner_details/scripts))
- Logs generated during training ([`logs/`](examiner_details/logs))

## License

This software and code is distributed under [MIT License](LICENSE)

## Citation

Tembe, M. Common local protein structural motifs supported by specific amino acids. Report [link](https://docs.google.com/document/d/1P0-RoaRTZxS-yPdZcdrNeWGD_X1I_7Ir/edit?usp=drive_link&ouid=107271459588602317511&rtpof=true&sd=true) (2024).

Lategan, F.A., Schreiber, C. & Patterton, H.G. SeqPredNN: a neural network that generates protein sequences that fold into specified tertiary structures. BMC Bioinformatics 24, 373 (2023). https://doi.org/10.1186/s12859-023-05498-4
