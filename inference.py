import argparse
import pathlib
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from autoencoder import Autoencoder
from visualiser import Visualiser
from constants import AMINO_ACID_INDICES

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Features:
    def __init__(self, translations, rotations, torsional_angles):
        self.translations = torch.tensor(translations, dtype=torch.float32)
        self.rotations = torch.tensor(rotations, dtype=torch.float32)
        self.torsional_angles = torch.tensor(torsional_angles, dtype=torch.float32)

    def get_feature_vector(self):
        translations_flat = self.translations.view(self.translations.size(0), -1)
        rotations_flat = self.rotations.view(self.rotations.size(0), -1)
        torsional_angles_flat = self.torsional_angles.view(self.torsional_angles.size(0), -1)
        return torch.cat([translations_flat, rotations_flat, torsional_angles_flat], dim=1)


class Residue:
    def __init__(self, features, label, chain_id, sequence_position):
        self.features = features
        self.label = label
        self.chain_id = chain_id
        self.sequence_position = sequence_position


class Chain:
    def __init__(self, chain_id, feature_data):
        self.chain_id = chain_id
        self.labels = torch.tensor(feature_data['residue_labels'], dtype=torch.long)
        self.features = Features(
            translations=feature_data['translations'],
            rotations=feature_data['rotations'],
            torsional_angles=feature_data['torsional_angles'],
        )

    def get_residues(self):
        labels = self.labels
        feature_vectors = self.features.get_feature_vector()
        return [
            Residue(
                features=feature_vectors[i],
                label=labels[i].item(),
                chain_id=self.chain_id,
                sequence_position=i,
            )
            for i in range(len(labels))
        ]


class StructureDataset(Dataset):
    def __init__(self, feature_directory, chain_list_file, scaler=None, seed=None):
        self.feature_directory = pathlib.Path(feature_directory)
        self.chain_ids = self._load_chain_ids(chain_list_file)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.residues = []
        self.chain_shapes = {}
        self._load_and_process_chains()

        if scaler:
            self._normalize_features(scaler)

        if not self.residues:
            raise ValueError("No valid data found after processing chains.")

    def _load_chain_ids(self, chain_list_file):
        with open(chain_list_file, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def _load_chain_features(self, chain_id):
        feature_file = self.feature_directory / f'{chain_id}.npz'
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file {feature_file} not found.")
        return dict(np.load(feature_file))

    def _process_chain(self, chain_id, feature_data):
        chain = Chain(chain_id, feature_data)
        residues = chain.get_residues()
        if residues:
            self.residues.extend(residues)
            self.chain_shapes[chain_id] = chain.features

    def _load_and_process_chains(self):
        for chain_id in self.chain_ids:
            chain_data = self._load_chain_features(chain_id)
            self._process_chain(chain_id, chain_data)

    def _normalize_features(self, scaler):
        all_features = torch.stack([residue.features for residue in self.residues])
        normalized_features = scaler.transform(all_features)
        for i, residue in enumerate(self.residues):
            residue.features = torch.tensor(normalized_features[i], dtype=torch.float32)

    def __len__(self):
        return len(self.residues)

    def __getitem__(self, idx):
        return self.residues[idx]


class AutoencoderInference:
    def __init__(
        self,
        input_directory,
        chain_list_file,
        model_path,
        output_directory,
        batch_size,
        device,
        save_latent_space_vectors=False,
        plot_feature_space=False,
        seed=None
    ):
        self.input_directory = pathlib.Path(input_directory)
        self.chain_list_file = chain_list_file
        self.model_path = pathlib.Path(model_path)
        self.output_directory = pathlib.Path(output_directory)
        self.batch_size = batch_size
        self.device = device
        self.save_latent_space_vectors = save_latent_space_vectors
        self.plot_feature_space = plot_feature_space
        self.seed = seed

        self._prepare_output_directory()
        self.model, self.scaler = self._load_model()
        self.dataset = self._load_dataset()
        self.data_loader = self._create_data_loader()
        self.visualizer = Visualiser(self.output_directory)

        logger.info("=== Chains ===\n")
        logger.info(f"Number of chains processed: {len(self.dataset.chain_ids)}\n")

        logger.info("=== Dataset Sizes ===\n")
        logger.info(f"Number of residues processed: {len(self.dataset.residues)}\n")

        logger.info("\n=== Model Architecture ===\n")
        logger.info(self.model)
        logger.info("\n")

    def _prepare_output_directory(self):
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        if not self.model_path.exists() or self.model_path.suffix != '.pth':
            raise FileNotFoundError(f"Model file {self.model_path} not found or is not a .pth file.")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        scaler = checkpoint['scaler']

        model = Autoencoder(
            input_dim=config['input_dim'],
            hidden_layers=config['Layers'],
            latent_dim=config['Latent Dimension'],
            dropout=config['Dropout Rate'],
            negative_slope=config['Negative Slope'],
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, scaler

    def _load_dataset(self):
        dataset = StructureDataset(self.input_directory, self.chain_list_file, seed=self.seed)
        dataset._normalize_features(self.scaler)
        return dataset

    def _create_data_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._residue_collate_fn,
        )

    def _residue_collate_fn(self, batch):
        return batch

    def inference(self):
        logger.info("Starting forward pass...\n")
        all_latent_vectors = []
        all_reconstructed_vectors = []
        all_residues = []

        mse_per_residue = {index: [] for index in AMINO_ACID_INDICES.values()}
        loss_fn = nn.MSELoss()

        self.model.eval()
        with torch.no_grad():
            for residues in self.data_loader:
                input_vectors = torch.stack([residue.features for residue in residues]).to(self.device)
                reconstructed_vectors, latent_vectors = self.model(input_vectors)

                loss = loss_fn(reconstructed_vectors, input_vectors)
                mse = loss.item()

                for residue in residues:
                    mse_per_residue[residue.label].append(mse)

                all_latent_vectors.append(latent_vectors.cpu().numpy())
                all_reconstructed_vectors.append(reconstructed_vectors.cpu().numpy())
                all_residues.extend(residues)

        mse_per_residue_avg = {
            self.visualizer._get_residue_name(index): np.mean(mse_list) if mse_list else 0.0
            for index, mse_list in mse_per_residue.items()
        }

        num_residues = len(self.dataset)

        logger.info("Inference completed.\n")
        logger.info("Generating plots and reports...\n")

        self._generate_plots_and_reports(
            all_residues,
            mse_per_residue_avg,
            num_residues,
            latent_vectors=np.concatenate(all_latent_vectors, axis=0),
            reconstructed_vectors=np.concatenate(all_reconstructed_vectors, axis=0)
        )

        logger.info("Forward pass processing completed successfully.\n")

    def _generate_plots_and_reports(self, all_residues, mse_per_residue_avg, num_residues, latent_vectors, reconstructed_vectors):
        pca_metrics_input = None
        umap_metrics_input = None

        if self.plot_feature_space:
            original_features = torch.stack([residue.features for residue in all_residues]).numpy()
            original_residue_labels = np.array([residue.label for residue in all_residues])

            pca_metrics_input = self.visualizer.plot_pca_projection(
                original_features,
                original_residue_labels,
                data_set_label='input features'
            )

            umap_metrics_input = self.visualizer.plot_umap_projection(
                original_features,
                original_residue_labels,
                data_set_label='input features'
            )

        umap_metrics_latent = self.visualizer.plot_umap_projection(
            latent_vectors,
            [residue.label for residue in all_residues],
            data_set_label='latent vectors',
        )

        self.visualizer.generate_inference_report(
            model=self.model,
            per_residue_mse=mse_per_residue_avg,
            num_residues=num_residues,
            num_chains=len(self.dataset.chain_ids),
            umap_metrics_latent=umap_metrics_latent,
            pca_metrics_input=pca_metrics_input,
            umap_metrics_input=umap_metrics_input
        )

        output_dir = self.output_directory / 'reconstructed_features'
        self.visualizer.save_features(
            all_residues,
            latent_vectors,
            reconstructed_vectors,
            self.dataset.chain_shapes,
            output_dir,
            scaler=self.scaler,
            save_latent_vectors=self.save_latent_space_vectors,
        )


def get_arguments():
    parser = argparse.ArgumentParser(description="Perform forward pass using a trained autoencoder model.")
    parser.add_argument("input_directory", type=str, help="Directory containing feature files.")
    parser.add_argument("chain_list_file", type=str, help="File containing chain IDs.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (.pth).")
    parser.add_argument("-o", "--output_directory", type=str, default="./inference", help="Directory to save the output.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing.")
    parser.add_argument("--save_latent_space_vectors", action="store_true", help="Save reconstructed features along with latent space vectors.")
    parser.add_argument("--plot_feature_space", action="store_true", help="Plots the PCA and UMAP projections of the input feature space.")
    parser.add_argument("--seed", type=int, default=42, help="Set the seed for reproducibility.")
    return parser.parse_args()


def inference():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder_inference = AutoencoderInference(
        input_directory=args.input_directory,
        chain_list_file=args.chain_list_file,
        model_path=args.model_path,
        output_directory=args.output_directory,
        batch_size=args.batch_size,
        device=device,
        save_latent_space_vectors=args.save_latent_space_vectors,
        plot_feature_space=args.plot_feature_space,
        seed=args.seed
    )
    autoencoder_inference.inference()


if __name__ == "__main__":
    inference()