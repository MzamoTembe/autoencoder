import pathlib
import logging
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from constants import AMINO_ACID_INDICES

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Visualiser:
    def __init__(self, output_directory):
        self.output_directory = pathlib.Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate_training_report(
        self, model, config, per_residue_mse, per_class_loss_history,
        train_losses, val_losses, optimizer, num_training_residues, num_validation_residues,
        dataset_size, balanced_sampling, num_test_chains, num_training_chains, umap_metrics=None
    ):
        report_file = self.output_directory / 'report.txt'
        with open(report_file, 'w') as f:
            f.write("=== Training Report ===\n\n")

            f.write("=== Chains ===\n")
            f.write(f"Number of test chains: {num_test_chains}\n")
            f.write(f"Number of training chains: {num_training_chains}\n")
            f.write(f"Total Chains: {num_test_chains + num_training_chains}\n")
            f.write("\n")

            f.write("=== Dataset Sizes ===\n")
            f.write(f"Original dataset size (unbalanced): {dataset_size}\n")
            if balanced_sampling:
                f.write(f"Number of training residues (balanced): {num_training_residues}\n")
            else:
                f.write(f"Number of training residues: {num_training_residues}\n")
            f.write(f"Number of validation residues: {num_validation_residues}\n")
            f.write(f"Total residues: {num_training_residues + num_validation_residues}\n")
            f.write("\n")

            f.write("=== Model Architecture ===\n")
            f.write(str(model))
            f.write("\n\n")

            f.write("=== Model Parameters ===\n")
            for param, value in config.items():
                f.write(f"{param}: {value}\n")
            f.write("\n")

            f.write("=== Optimizer ===\n")
            f.write(f"{optimizer}\n")
            f.write("\n")

            f.write("=== Per-residue MSE ===\n")
            for residue_index, mse in per_residue_mse.items():
                residue = self._get_residue_name(residue_index)
                f.write(f"{residue}: {mse:.6f}\n")
            f.write("\n")

            f.write("=== Training vs Validation Loss ===\n")
            for epoch in range(len(train_losses)):
                f.write(
                    f"Epoch {epoch+1}: Training Loss: {train_losses[epoch]:.6f}, Validation Loss: {val_losses[epoch]:.6f}\n"
                )
            f.write("\n")

            if umap_metrics:
                f.write("=== UMAP Metrics ===\n")
                f.write(f"Neighbours: {umap_metrics['n_neighbors']}\n")
                f.write(f"min_dist: {umap_metrics['min_dist']}\n")
                f.write("\n")

    def generate_inference_report(self, model, per_residue_mse, num_residues, num_chains, umap_metrics_latent, pca_metrics_input=None, umap_metrics_input=None):
        report_file = self.output_directory / 'report.txt'
        with open(report_file, 'w') as f:
            f.write("=== Inference Report ===\n\n")

            f.write("=== Chains ===\n")
            f.write(f"Number of chains processed: {num_chains}\n")
            f.write("\n")

            f.write("=== Dataset Size ===\n")
            f.write(f"Number of residues processed: {num_residues}\n")
            f.write("\n")

            f.write("=== Model Architecture ===\n")
            f.write(str(model))
            f.write("\n\n")

            f.write("=== Per-residue MSE ===\n")
            for residue, mse in per_residue_mse.items():
                f.write(f"{residue}: {mse:.6f}\n")
            f.write("\n")

            if pca_metrics_input:
                f.write("=== PCA Metrics (Input) ===\n")
                f.write(f"Explained Variance Ratios: {pca_metrics_input['explained_variance_ratio']}\n")
                f.write(f"Cumulative Explained Variance Ratio: {pca_metrics_input['cumulative_variance_ratio']:.2f}\n")
                f.write("\n")

            if umap_metrics_input:
                f.write("=== UMAP Metrics (Input) ===\n")
                f.write(f"Neighbours: {umap_metrics_input['n_neighbors']}\n")
                f.write(f"min_dist: {umap_metrics_input['min_dist']}\n")
                f.write("\n")

            if umap_metrics_latent:
                f.write("=== UMAP Metrics (Latent) ===\n")
                f.write(f"Neighbours: {umap_metrics_latent['n_neighbors']}\n")
                f.write(f"min_dist: {umap_metrics_latent['min_dist']}\n")
                f.write("\n")

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
        plt.savefig(self.output_directory / 'loss_curve.png')
        plt.close()

        loss_df = pd.DataFrame({
            'Epoch': epochs,
            'Training Loss': train_losses,
            'Validation Loss': val_losses,
        })
        loss_df.to_csv(self.output_directory / 'loss_curve.csv', index=False)

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
        plt.savefig(self.output_directory / 'per_class_loss.png')
        plt.close()

        data = {'Epoch': epochs}
        for residue in AMINO_ACID_INDICES.keys():
            data[f'Train {residue}'] = train_history[residue]
            data[f'Val {residue}'] = val_history[residue]
        df = pd.DataFrame(data)
        df.to_csv(self.output_directory / 'per_class_loss.csv', index=False)

    def plot_pca_projection(self, features, residue_labels, data_set_label='', n_components=2):
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        logger.info(f'\nPCA Explained Variance Ratios (first {n_components} components): {explained_variance_ratio}')
        logger.info(f'PCA Cumulative Explained Variance Ratio (first {n_components} components): {cumulative_variance_ratio[-1]:.2f}\n')

        residue_names = [self._get_residue_name(label) for label in residue_labels]
        df = pd.DataFrame({
            'PCA1': pca_features[:, 0],
            'PCA2': pca_features[:, 1],
            'Residue': residue_names,
        })
        plt.figure(figsize=(10, 8))
        unique_residues = df['Residue'].unique()
        colors = plt.cm.get_cmap('tab20', len(unique_residues))
        for i, residue in enumerate(unique_residues):
            subset = df[df['Residue'] == residue]
            plt.scatter(subset['PCA1'], subset['PCA2'], label=residue, color=colors(i % 20), s=10)
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.title(f'PCA Projection {f"({data_set_label})" if data_set_label else ""}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        plt.tight_layout()
        plt.savefig(self.output_directory / 'pca_projection.png')
        plt.close()
        df.to_csv(self.output_directory / 'pca_projection.csv', index=False)

        return {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio[-1]
        }

    def plot_umap_projection(self, features, residue_labels, data_set_label='', n_components=2, n_neighbors=16, min_dist=0.1, metric='euclidean', random_state=42):
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        umap_features = reducer.fit_transform(features)

        residue_names = [self._get_residue_name(label) for label in residue_labels]
        df = pd.DataFrame({
            'UMAP1': umap_features[:, 0],
            'UMAP2': umap_features[:, 1],
            'Residue': residue_names,
        })
        plt.figure(figsize=(10, 8))
        unique_residues = df['Residue'].unique()
        colors = plt.cm.get_cmap('tab20', len(unique_residues))
        for i, residue in enumerate(unique_residues):
            subset = df[df['Residue'] == residue]
            plt.scatter(subset['UMAP1'], subset['UMAP2'], label=residue, color=colors(i % 20), s=10)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'UMAP Projection {f"({data_set_label})" if data_set_label else ""}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        plt.tight_layout()
        plt.savefig(self.output_directory / f'umap_projection_{data_set_label.replace(" ", "_")}.png')
        plt.close()
        df.to_csv(self.output_directory / f'umap_projection_{data_set_label.replace(" ", "_")}.csv', index=False)

        return {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }

    def save_features(self, all_residues, latent_vectors, reconstructed_vectors, chain_shapes, output_dir, scaler=None, save_latent_vectors=True):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chain_to_residues = {}
        for idx, residue in enumerate(all_residues):
            chain_to_residues.setdefault(residue.chain_id, []).append((idx, residue))

        for chain_id, residue_data in chain_to_residues.items():
            indices, residues = zip(*sorted(residue_data, key=lambda x: x[1].sequence_position))
            indices = np.array(indices)
            residue_labels = np.array([residue.label for residue in residues])

            chain_reconstructed_vectors = reconstructed_vectors[indices]
            original_reconstructed_vectors = scaler.inverse_transform(chain_reconstructed_vectors)

            chain_shape = chain_shapes[chain_id]
            translations_dim = np.prod(chain_shape.translations.shape[1:])
            rotations_dim = np.prod(chain_shape.rotations.shape[1:])
            translations = original_reconstructed_vectors[:, :translations_dim].reshape((-1, *chain_shape.translations.shape[1:]))
            rotations = original_reconstructed_vectors[:, translations_dim:translations_dim + rotations_dim].reshape((-1, *chain_shape.rotations.shape[1:]))
            torsional_angles = original_reconstructed_vectors[:, translations_dim + rotations_dim:].reshape((-1, *chain_shape.torsional_angles.shape[1:]))

            save_dict = {
                'residue_labels': residue_labels,
                'translations': translations,
                'rotations': rotations,
                'torsional_angles': torsional_angles,
            }
            if save_latent_vectors and latent_vectors is not None:
                save_dict['latent_vectors'] = latent_vectors[indices]

            np.savez(output_dir / f"{chain_id}.npz", **save_dict)

        with open(output_dir / "reconstructed_chain_list.txt", 'w') as f:
            f.writelines(f"{chain_id}\n" for chain_id in chain_to_residues)

        logger.info(f"Saved features for {len(chain_to_residues)} chains")

    def _get_residue_name(self, label):
        for amino_acid, index in AMINO_ACID_INDICES.items():
            if index == label:
                return amino_acid
        return 'Unknown'