import pathlib
import logging
import umap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import AMINO_ACID_INDICES

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Visualiser:
    def __init__(self, save_path=None, reconstruction_threshold=0.1):
        self.save_path = pathlib.Path(save_path) if save_path else pathlib.Path.cwd()
        self.reconstruction_threshold = reconstruction_threshold

    def generate_training_report(self, model, per_residue_mse,
                                 per_class_loss_history, train_losses, val_losses,
                                 optimizer, num_training_residues, num_validation_residues):
        report_file = self.save_path / 'model_report.txt'
        with open(report_file, 'w') as f:
            f.write("=== Model Architecture ===\n")
            f.write(str(model))
            f.write("\n\n=== Model Parameters ===\n")
            for name, param in model.named_parameters():
                f.write(f"{name}: {param.size()}\n")
            f.write("\n=== Optimizer ===\n")
            f.write(f"{optimizer}\n")
            f.write("\n=== Dataset Sizes ===\n")
            f.write(f"Number of training residues: {num_training_residues}\n")
            f.write(f"Number of validation residues: {num_validation_residues}\n")
            f.write(f"Total residues: {num_training_residues + num_validation_residues}\n")
            f.write("\n=== Per-residue MSE ===\n")
            for residue_index, mse in per_residue_mse.items():
                residue = self._get_residue_name(residue_index)
                f.write(f"{residue}: {mse:.6f}\n")
            f.write("\n=== Training vs Validation Loss ===\n")
            for epoch in range(len(train_losses)):
                f.write(f"Epoch {epoch+1}: Training Loss: {train_losses[epoch]:.6f}, Validation Loss: {val_losses[epoch]:.6f}\n")

    def generate_forward_pass_report(self, model, per_residue_mse, num_residues):
        report_file = self.save_path / 'report.txt'
        with open(report_file, 'w') as f:
            f.write("=== Model Architecture ===\n")
            f.write(str(model))
            f.write("\n\n=== Dataset Size ===\n")
            f.write(f"Number of residues processed: {num_residues}\n")
            f.write("\n=== Per-residue MSE (Forward Pass) ===\n")
            for residue, mse in per_residue_mse.items():
                f.write(f"{residue}: {mse:.6f}\n")

    def plot_loss_curves(self, train_losses, val_losses):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Training vs. Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path / 'loss_curve.png')
        plt.close()

        loss_df = pd.DataFrame({
            'Epoch': epochs,
            'Training Loss': train_losses,
            'Validation Loss': val_losses
        })
        loss_df.to_csv(self.save_path / 'loss_curve.csv', index=False)

    def plot_per_class_loss_over_epochs(self, train_history, val_history):
        epochs = range(1, len(next(iter(train_history.values()))) + 1)
        plt.figure(figsize=(14, 8))
        for residue in AMINO_ACID_INDICES.keys():
            train_losses = train_history[residue]
            val_losses = val_history[residue]
            plt.plot(epochs, train_losses, label=f'Train {residue}')
            plt.plot(epochs, val_losses, label=f'Val {residue}', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Per-Class Loss Over Epochs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_path / 'per_class_loss_over_epochs.png')
        plt.close()

        data = {'Epoch': epochs}
        for residue in AMINO_ACID_INDICES.keys():
            data[f'Train {residue}'] = train_history[residue]
            data[f'Val {residue}'] = val_history[residue]
        df = pd.DataFrame(data)
        df.to_csv(self.save_path / 'per_class_loss_over_epochs.csv', index=False)

    def plot_reconstruction_error_distribution(self, reconstruction_errors, residue_labels, data_set_label=''):
        error_df = pd.DataFrame({
            'Error': reconstruction_errors,
            'Residue': [self._get_residue_name(label) for label in residue_labels]
        })
        plt.figure(figsize=(12, 8))
        residues = error_df['Residue'].unique()
        positions = range(len(residues))

        for i, residue in enumerate(residues):
            errors = error_df[error_df['Residue'] == residue]['Error']
            jittered_positions = np.random.normal(i, 0.05, size=len(errors))
            plt.scatter(jittered_positions, errors, alpha=0.6, label=residue)

        plt.title(f'Reconstruction Error Distribution by Residue Type ({data_set_label})')
        plt.xlabel('Residue Type')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.xticks(positions, residues, rotation=90)
        plt.tight_layout()
        plt.savefig(self.save_path / f'reconstruction_error_distribution_{data_set_label.replace(" ", "_")}.png')
        plt.close()

        error_df.to_csv(self.save_path / f'reconstruction_error_distribution_{data_set_label.replace(" ", "_")}.csv',
                        index=False)

    def plot_latent_space(self, latent_vectors, residue_labels, data_set_label=''):
        reducer = umap.UMAP()
        latent_2d = reducer.fit_transform(latent_vectors)
        residue_names = [self._get_residue_name(label) for label in residue_labels]
        df = pd.DataFrame({
            'UMAP1': latent_2d[:, 0],
            'UMAP2': latent_2d[:, 1],
            'Residue': residue_names
        })
        plt.figure(figsize=(10, 8))
        unique_residues = df['Residue'].unique()
        for residue in unique_residues:
            subset = df[df['Residue'] == residue]
            plt.scatter(subset['UMAP1'], subset['UMAP2'], label=residue, s=10)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title(f'Latent Space Visualization (UMAP) ({data_set_label})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        plt.tight_layout()
        plt.savefig(self.save_path / f'latent_space_{data_set_label.replace(" ", "_")}.png')
        plt.close()
        df.to_csv(self.save_path / f'latent_space_{data_set_label.replace(" ", "_")}.csv', index=False)

    def save_features(self, all_residues, latent_vectors, reconstructed_vectors, chain_shapes, output_dir, save_latent_vectors=True):
        output_dir.mkdir(parents=True, exist_ok=True)
        chain_to_residues = {}
        for idx, residue in enumerate(all_residues):
            chain_to_residues.setdefault(residue.chain_id, []).append((idx, residue))

        for chain_id, residue_data in chain_to_residues.items():
            residue_data_sorted = sorted(residue_data, key=lambda x: x[1].sequence_position)
            indices, residues = zip(*residue_data_sorted)
            indices = np.array(indices)

            labels = np.array([residue.label for residue in residues])
            recons = reconstructed_vectors[indices]
            translations, rotations, torsional_angles = self._extract_features_from_vector(
                torch.tensor(recons), chain_id, chain_shapes
            )
            save_dict = {
                'residue_labels': labels,
                'translations': translations.numpy(),
                'rotations': rotations.numpy(),
                'torsional_angles': torsional_angles.numpy()
            }
            if save_latent_vectors and latent_vectors is not None:
                latents = latent_vectors[indices]
                save_dict['latent_vectors'] = latents
            np.savez(output_dir / f"{chain_id}.npz", **save_dict)

        with open(output_dir / "reconstructed_chain_list.txt", 'w') as f:
            for chain_id in chain_to_residues.keys():
                f.write(f"{chain_id}\n")

    def _extract_features_from_vector(self, vector, chain_id, chain_shapes):
        features = chain_shapes[chain_id]
        batch_size = vector.size(0)
        translations_dim = features.translations.shape[1] * features.translations.shape[2]
        rotations_dim = features.rotations.shape[1] * features.rotations.shape[2]

        translations_shape = features.translations.shape[1:]
        rotations_shape = features.rotations.shape[1:]
        torsional_angles_shape = features.torsional_angles.shape[1:]

        translations = vector[:, :translations_dim].reshape(batch_size, *translations_shape)
        rotations = vector[:, translations_dim:translations_dim + rotations_dim].reshape(batch_size, *rotations_shape)
        torsional_angles = vector[:, translations_dim + rotations_dim:].reshape(batch_size, *torsional_angles_shape)

        return translations, rotations, torsional_angles

    def _get_residue_name(self, label):
        for amino_acid, index in AMINO_ACID_INDICES.items():
            if index == label:
                return amino_acid
        return 'Unknown'