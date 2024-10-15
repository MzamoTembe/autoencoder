import logging
import argparse
import pathlib
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
from autoencoder import Autoencoder
from visualiser import Visualiser
from constants import AMINO_ACID_INDICES, STANDARD_AMINO_ACIDS
from collections import Counter

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

    def get_valid_residues(self):
        valid_indices = self.labels != AMINO_ACID_INDICES['X']
        labels = self.labels[valid_indices]
        feature_vectors = self.features.get_feature_vector()[valid_indices]
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
    def __init__(self, feature_directory, chain_list_file, test_chain_list=None, seed=None):
        self.feature_directory = pathlib.Path(feature_directory)
        self.chain_ids = self._load_chain_ids(chain_list_file)
        if test_chain_list:
            test_chains = self._load_chain_ids(test_chain_list)
            self.chain_ids = [chain for chain in self.chain_ids if chain not in test_chains]
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.residues = []
        self.chains = []
        self.chain_shapes = {}
        self._load_and_process_chains()
        self._normalize_features()

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
        residues = chain.get_valid_residues()
        if residues:
            self.residues.extend(residues)
            self.chains.append(chain)
            self.chain_shapes[chain_id] = chain.features

    def _load_and_process_chains(self):
        for chain_id in self.chain_ids:
            chain_data = self._load_chain_features(chain_id)
            self._process_chain(chain_id, chain_data)

    def _normalize_features(self):
        all_features = torch.stack([residue.features for residue in self.residues])
        self.scaler = StandardScaler()
        normalized_features = self.scaler.fit_transform(all_features)
        for i, residue in enumerate(self.residues):
            residue.features = torch.tensor(normalized_features[i], dtype=torch.float32)

    def __len__(self):
        return len(self.residues)

    def __getitem__(self, idx):
        return self.residues[idx]


class AutoencoderTrainer:
    def __init__(
        self,
        input_directory,
        chain_list_file,
        test_chain_list,
        output_directory,
        layers,
        latent_dim,
        dropout,
        batch_size,
        learning_rate,
        epochs,
        device,
        balanced_sampling=False,
        seed=None,
        negative_slope=0.01,
        save_val_features=False,
        train_val_split=0.8,
    ):
        self.input_directory = pathlib.Path(input_directory)
        self.output_directory = pathlib.Path(output_directory)
        self.layers = layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.balanced_sampling = balanced_sampling
        self.seed = seed
        self.negative_slope = negative_slope
        self.save_val_features = save_val_features
        self.train_val_split = train_val_split
        self.random_number_generator = np.random.default_rng(seed=seed)

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.dataset = StructureDataset(self.input_directory, chain_list_file, test_chain_list, seed=self.seed)
        self.dataset_size = len(self.dataset)
        self.sample_space = np.arange(self.dataset_size)

        self.train_loader, self.val_loader = self._split_dataset()

        sample_residues = next(iter(self.train_loader))
        sample_features = sample_residues[0].features
        self.input_dim = sample_features.shape[0]

        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_layers=self.layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            negative_slope=self.negative_slope,
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.visualizer = Visualiser(self.output_directory)

        logger.info("\n=== Model Architecture ===\n")
        logger.info(self.model)
        logger.info("\n")

    def _split_dataset(self):
        if self.balanced_sampling:
            train_dataset, val_dataset = self.balanced()
        else:
            train_dataset, val_dataset = self.unbalanced()

        self.num_training_residues = len(train_dataset)
        self.num_validation_residues = len(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        return train_loader, val_loader

    def randomize_validation_set(self):
        sample_size = int(self.dataset_size * self.train_val_split)
        train_indices = self.random_number_generator.choice(self.sample_space, size=sample_size, replace=False)
        validation_indices = np.setdiff1d(self.sample_space, train_indices, assume_unique=True)
        return train_indices, validation_indices

    def unbalanced(self):
        train_indices, validation_indices = self.randomize_validation_set()
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, validation_indices)
        return train_dataset, val_dataset

    def balanced(self):
        train_indices, validation_indices = self.randomize_validation_set()
        non_validation_labels = np.array([self.dataset[i].label for i in train_indices])
        class_indices = [
            np.extract(non_validation_labels == amino_acid, train_indices)
            for amino_acid in range(len(AMINO_ACID_INDICES) - 1)
        ]
        residue_frequencies = [len(amino_acid_partition) for amino_acid_partition in class_indices]
        class_sample_size = min(residue_frequencies)
        class_samples = [
            self.random_number_generator.choice(indices, size=class_sample_size, replace=False)
            for indices in class_indices
        ]
        balanced_train_indices = np.concatenate(class_samples, axis=0)
        train_dataset = Subset(self.dataset, balanced_train_indices)
        val_dataset = Subset(self.dataset, validation_indices)
        amino_acid_counts = Counter([self.dataset[i].label for i in balanced_train_indices])
        print("Amino acid counts in balanced training set:")
        for amino_acid, count in amino_acid_counts.items():
            print(f"{STANDARD_AMINO_ACIDS[amino_acid]}: {count}")
        return train_dataset, val_dataset

    def _collate_fn(self, batch):
        return batch

    def train(self):
        logger.info("Starting training loop...\n")
        train_losses = []
        val_losses = []
        per_residue_train_history = {residue: [] for residue in AMINO_ACID_INDICES.keys()}
        per_residue_val_history = {residue: [] for residue in AMINO_ACID_INDICES.keys()}

        logger.info(f"Number of training residues: {self.num_training_residues}")
        logger.info(f"Number of validation residues: {self.num_validation_residues}\n")

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}\n")
            train_loss, train_mse_per_residue = self._train_epoch()
            val_results = self._validate_epoch()
            val_loss = val_results['avg_loss']
            val_mse_per_residue = val_results['mse_per_residue']

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            for residue_name, residue_index in AMINO_ACID_INDICES.items():
                per_residue_train_history[residue_name].append(train_mse_per_residue.get(residue_index, 0.0))
                per_residue_val_history[residue_name].append(val_mse_per_residue.get(residue_index, 0.0))

            logger.info(f"Training Loss: {train_loss:.6f}")
            logger.info(f"Validation Loss: {val_loss:.6f}\n")

        logger.info("Training loop completed.\n")

        config = self._collect_config()

        self._save_model()
        self._generate_plots_and_reports(train_losses, val_losses, per_residue_train_history, per_residue_val_history, val_results, config)

        logger.info(f"Final Validation MSE: {val_losses[-1]:.6f}\n")
        logger.info("Training process completed successfully.\n")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        mse_per_residue = {index: [] for index in AMINO_ACID_INDICES.values()}

        for residues in self.train_loader:
            input_vectors = torch.stack([residue.features for residue in residues]).to(self.device)
            reconstructed_vectors, _ = self.model(input_vectors)
            loss = self.loss_fn(reconstructed_vectors, input_vectors)
            total_loss += loss.item()

            mse = loss.item()
            for residue in residues:
                mse_per_residue[residue.label].append(mse)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.train_loader)
        mse_per_residue_avg = {
            index: np.mean(mse_list) if mse_list else 0.0
            for index, mse_list in mse_per_residue.items()
        }
        return avg_loss, mse_per_residue_avg

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        mse_per_residue = {index: [] for index in AMINO_ACID_INDICES.values()}
        all_latents = []
        all_reconstructed_vectors = []
        all_residues = []
        all_reconstruction_errors = []

        with torch.no_grad():
            for residues in self.val_loader:
                input_vectors = torch.stack([residue.features for residue in residues]).to(self.device)
                reconstructed_vectors, latents = self.model(input_vectors)
                loss = self.loss_fn(reconstructed_vectors, input_vectors)
                total_loss += loss.item()

                mse = loss.item()
                for residue in residues:
                    mse_per_residue[residue.label].append(mse)
                    all_reconstruction_errors.append(mse)

                all_latents.append(latents.cpu().numpy())
                all_reconstructed_vectors.append(reconstructed_vectors.cpu().numpy())
                all_residues.extend(residues)

        avg_loss = total_loss / len(self.val_loader)
        mse_per_residue_avg = {
            index: np.mean(mse_list) if mse_list else 0.0
            for index, mse_list in mse_per_residue.items()
        }
        all_latents = np.concatenate(all_latents, axis=0)
        all_reconstructed_vectors = np.concatenate(all_reconstructed_vectors, axis=0)

        return {
            'avg_loss': avg_loss,
            'mse_per_residue': mse_per_residue_avg,
            'latent_vectors': all_latents,
            'reconstructed_vectors': all_reconstructed_vectors,
            'residues': all_residues,
            'reconstruction_errors': np.array(all_reconstruction_errors),
        }

    def _generate_plots_and_reports(self, train_losses, val_losses, per_residue_train_history, per_residue_val_history, val_results, config):
        logger.info("Generating plots and reports...\n")
        self.visualizer.plot_loss_curves(train_losses, val_losses)
        self.visualizer.plot_per_class_loss_over_epochs(per_residue_train_history, per_residue_val_history)
        self.visualizer.generate_training_report(
            model=self.model,
            config=config,
            per_residue_mse=val_results['mse_per_residue'],
            per_class_loss_history=per_residue_val_history,
            train_losses=train_losses,
            val_losses=val_losses,
            optimizer=self.optimizer,
            num_training_residues=self.num_training_residues,
            num_validation_residues=self.num_validation_residues,
        )
        self.visualizer.plot_reconstruction_error_distribution(
            val_results['reconstruction_errors'],
            [residue.label for residue in val_results['residues']],
            data_set_label='Validation Set',
        )
        self.visualizer.plot_latent_space(
            val_results['latent_vectors'],
            [residue.label for residue in val_results['residues']],
            data_set_label='Validation Set',
        )

        if self.save_val_features:
            self.visualizer.save_features(
                val_results['residues'],
                val_results['latent_vectors'],
                val_results['reconstructed_vectors'],
                self.dataset.chain_shapes,
                self.output_directory / 'validation_features',
                scaler=self.dataset.scaler,
                save_latent_vectors=self.save_val_features,
            )

    def _save_model(self):
        model_save_path = self.output_directory / 'trained_model.pth'
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': self._collect_config(),
                'scaler': self.dataset.scaler,
            },
            model_save_path,
        )
        logger.info(f"Model saved to {model_save_path}\n")

    def _collect_config(self):
        return {
            'input_dim': self.input_dim,
            'Layers': self.layers,
            'Latent Dimension': self.latent_dim,
            'Dropout Rate': self.dropout,
            'Batch Size': self.batch_size,
            'Learning Rate': self.learning_rate,
            'Epochs': self.epochs,
            'Device': self.device,
            'Balanced Sampling': self.balanced_sampling,
            'Seed': self.seed,
            'Negative Slope': self.negative_slope,
            'Save Validation Features': self.save_val_features,
            'Train-Validation Split': self.train_val_split,
        }


def get_arguments():
    parser = argparse.ArgumentParser(description="Train an autoencoder model for amino acid features.")
    parser.add_argument("input_directory", type=str, help="Directory containing feature files.")
    parser.add_argument("chain_list_file", type=str, help="File containing chain IDs (chain_list.txt).")
    parser.add_argument("test_chain_list", type=str, help="File containing chain IDs to be excluded from training (test_chain_list.txt).")
    parser.add_argument("-o", "--output_directory", type=str, default="./model", help="Directory to save the model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--layers", type=lambda s: [int(x) for x in s.split(',')], default=[128, 64, 32], help="List of hidden layer sizes separated by commas.")
    parser.add_argument("--latent_dim", type=int, default=16, help="Size of the latent dimension.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--balanced_sampling", action="store_true", help="Use balanced sampling for residues.")
    parser.add_argument("--seed", type=int, default=42, help="Set the seed for random number generation.")
    parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU activation.")
    parser.add_argument("--save_val_features", action="store_true", help="Save features for the validation set.")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="Ratio for splitting the dataset.")
    return parser.parse_args()


def train_model():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = AutoencoderTrainer(
        input_directory=args.input_directory,
        chain_list_file=args.chain_list_file,
        test_chain_list=args.test_chain_list,
        output_directory=args.output_directory,
        layers=args.layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
        balanced_sampling=args.balanced_sampling,
        negative_slope=args.negative_slope,
        save_val_features=args.save_val_features,
        train_val_split=args.train_val_split,
    )
    trainer.train()


if __name__ == "__main__":
    train_model()