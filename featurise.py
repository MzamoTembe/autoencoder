import argparse
import gzip
import pathlib
import numpy as np
from Bio.PDB import PDBParser, Polypeptide
from scipy.spatial.transform import Rotation
from constants import AMINO_ACID_INDICES


class Features:
    def __init__(self, chain, num_neighbours):
        self.chain = chain
        self.num_neighbours = num_neighbours
        self.translations = None
        self.rotations = None
        self.torsional_angles = None

    def extract_features(self):
        if len(self.chain.residues) == 0:
            return None
        self.calculate_neighbour_indices()
        self.get_neighbour_data()
        self.calculate_basis_vectors()
        self.calculate_local_translations()
        self.calculate_rotations()
        self.encode_torsional_angles()
        return self

    def calculate_neighbour_indices(self):
        distances = np.linalg.norm(
            self.chain.ca_coords[:, None, :] - self.chain.ca_coords[None, :, :], axis=-1)
        self.neighbour_indices = np.argsort(distances, axis=1)[:, :self.num_neighbours + 1]

    def get_neighbour_data(self):
        indices = self.neighbour_indices
        self.neighbouring_residues = self.chain.residues[indices]
        self.neighbour_ca_coords = self.chain.ca_coords[indices]
        self.xyz_translations = self.neighbour_ca_coords - self.chain.ca_coords[:, None, :]
        self.phi_neighbours = self.chain.phi_angles[indices]
        self.psi_neighbours = self.chain.psi_angles[indices]

    def calculate_basis_vectors(self):
        n_coords = np.array([[res["N"].get_coord() for res in group]
                             for group in self.neighbouring_residues])
        c_coords = np.array([[res["C"].get_coord() for res in group]
                             for group in self.neighbouring_residues])
        u = self.normalize_vectors(c_coords - n_coords)
        n_to_ca = self.normalize_vectors(self.neighbour_ca_coords - n_coords)
        projections = self.project_vectors(n_to_ca, u) * u
        v = self.normalize_vectors(n_to_ca - projections)
        w = np.cross(u, v)
        self.basis_vectors = np.stack([u, v, w], axis=2)

    def calculate_local_translations(self):
        local_basis = self.basis_vectors[:, 0:1, :, :]
        translations = np.sum(
            self.xyz_translations[:, 1:, None, :] * local_basis, axis=-1)
        if translations.shape[1] < self.num_neighbours:
            pad_width = self.num_neighbours - translations.shape[1]
            translations = np.pad(
                translations, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        self.translations = translations

    def calculate_rotations(self):
        local_basis = self.basis_vectors[:, 0:1, :, :]
        neighbour_bases = self.basis_vectors[:, 1:, :, :]
        rotation_matrices = np.matmul(neighbour_bases, np.transpose(local_basis, (0, 1, 3, 2)))
        rotation_matrices = rotation_matrices.reshape(-1, 3, 3)
        rotations = Rotation.from_matrix(rotation_matrices).as_quat().reshape(
            len(self.chain.residues), -1, 4)
        if rotations.shape[1] < self.num_neighbours:
            pad_width = self.num_neighbours - rotations.shape[1]
            rotations = np.pad(
                rotations, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        self.rotations = rotations

    def encode_torsional_angles(self):
        phi_sin_cos = np.stack(
            [np.sin(self.phi_neighbours), np.cos(self.phi_neighbours)], axis=-1)
        psi_sin_cos = np.stack(
            [np.sin(self.psi_neighbours), np.cos(self.psi_neighbours)], axis=-1)
        torsional_angles = np.stack([phi_sin_cos, psi_sin_cos], axis=-1)
        expected_shape = self.num_neighbours + 1
        if torsional_angles.shape[1] < expected_shape:
            pad_width = expected_shape - torsional_angles.shape[1]
            torsional_angles = np.pad(
                torsional_angles, ((0, 0), (0, pad_width), (0, 0), (0, 0)), mode='constant')
        self.torsional_angles = torsional_angles

    def normalize_vectors(vectors):
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def project_vectors(vector_a, vector_b):
        return np.sum(vector_a * vector_b, axis=-1, keepdims=True) / np.sum(
            vector_b * vector_b, axis=-1, keepdims=True)


class Chain:
    def __init__(self, chain_id, chain, include_modified, num_neighbours, modified_residues):
        self.chain_id = chain_id
        self.include_modified = include_modified
        self.modified_residues = modified_residues
        self.residues = []
        self.ca_coords = []
        self.phi_angles = []
        self.psi_angles = []
        self.labels = []
        self.num_neighbours = num_neighbours
        self.extract_residues(chain)
        self.features = Features(self, num_neighbours).extract_features()

    def extract_residues(self, chain):
        ppb = Polypeptide.PPBuilder()
        for peptide in ppb.build_peptides(chain, aa_only=False):
            torsionals = peptide.get_phi_psi_list()
            for res, angles in zip(peptide, torsionals):
                if any(atom not in res for atom in ("N", "C", "CA")):
                    continue
                resname = self.process_residue(res)
                self.residues.append(res)
                self.ca_coords.append(res["CA"].get_coord())
                self.phi_angles.append(angles[0] if angles[0] is not None else 0.0)
                self.psi_angles.append(angles[1] if angles[1] is not None else 0.0)
                self.labels.append(AMINO_ACID_INDICES.get(resname, -1))
        self.ca_coords = np.array(self.ca_coords, dtype=float)
        self.phi_angles = np.array(self.phi_angles)
        self.psi_angles = np.array(self.psi_angles)
        self.labels = np.array(self.labels)
        self.residues = np.array(self.residues, dtype=object)

    def process_residue(self, residue):
        resname = residue.get_resname()
        if not Polypeptide.is_aa(residue, standard=True):
            if self.include_modified and resname in self.modified_residues:
                residue.resname = self.modified_residues[resname]
                resname = residue.resname
            else:
                residue.resname = 'X'
                resname = 'X'
        return resname


def load_chain_list(chain_list_file, pdb_dir):
    protein_dict = {}
    with open(chain_list_file, 'r') as f:
        next(f)
        for line in f:
            protein_name, pdb_file_path, chain_id = line.strip().split(',')
            pdb_file = pathlib.Path(pdb_dir) / pdb_file_path
            if protein_name not in protein_dict:
                protein_dict[protein_name] = (pdb_file, [chain_id])
            else:
                protein_dict[protein_name][1].append(chain_id)
    return protein_dict


def parse_modified_residues(pdb_file, include_modified):
    modified_residues = {}
    if include_modified:
        pdb_file.seek(0)
        for line in pdb_file:
            if line.startswith("MODRES"):
                mod_res = line[12:15].strip()
                std_res = line[24:27].strip()
                modified_residues[mod_res] = std_res
        pdb_file.seek(0)
    return modified_residues


def save_features(protein_id, chain_id, output_dir, labels, features):
    np.savez(output_dir / f"{protein_id}{chain_id}.npz",
             residue_labels=labels,
             translations=features.translations,
             rotations=features.rotations,
             torsional_angles=features.torsional_angles)
    with open(output_dir / "chain_list.txt", "a") as file:
        file.write(f"{protein_id}{chain_id}\n")


def process_proteins(protein_dict, args, output_dir):
    total_chains = sum(len(chains) for chains in protein_dict.values())
    chain_counter = 1
    for protein_id, (pdb_file_path, chain_ids) in protein_dict.items():
        if not pdb_file_path.exists():
            continue
        try:
            with gzip.open(pdb_file_path, 'rt') if args.gzip else open(pdb_file_path, 'r') as pdb_file:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(protein_id, pdb_file)
                modified_residues = parse_modified_residues(pdb_file, args.mod)
        except Exception as e:
            print(f"Error parsing PDB file {pdb_file_path}: {e}")
            continue
        for model in structure:
            for chain in model:
                if chain.id not in chain_ids:
                    continue
                print(f"Processing chain {chain_counter}/{total_chains}")
                chain_counter += 1
                chain_obj = Chain(chain.id, chain, args.mod, args.neighbours, modified_residues)
                labels = chain_obj.labels
                features = chain_obj.features
                if features is not None:
                    save_features(protein_id, chain.id, output_dir, labels, features)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract structural features from PDB files.")
    parser.add_argument("chain_list", type=str, help="CSV file listing protein chains to process.")
    parser.add_argument("pdb_dir", type=str, help="Directory containing PDB files.")
    parser.add_argument("-o", "--output_dir", type=str, default="./features", help="Output directory.")
    parser.add_argument("-g", "--gzip", action="store_true", help="Unzip gzipped PDB files.")
    parser.add_argument("-m", "--mod", action="store_true", help="Parse modified residues as unmodified amino acids.")
    parser.add_argument("-n", "--neighbours", type=int, default=16, help="Number of neighbouring residues.")
    return parser.parse_args()


def featurise():
    args = parse_args()
    chain_list = pathlib.Path(args.chain_list)
    pdb_dir = pathlib.Path(args.pdb_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    protein_dict = load_chain_list(chain_list, pdb_dir)
    process_proteins(protein_dict, args, output_dir)


if __name__ == "__main__":
    featurise()