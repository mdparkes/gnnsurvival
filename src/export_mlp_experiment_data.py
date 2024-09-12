import argparse
import io
import numpy as np
import os
import pickle
import re
import torch

from collections import OrderedDict
from filelock import FileLock
from torch import Tensor
from typing import Callable, Optional, Sequence

from dataset_definitions import CancerDataset
from models import SparseMLP, NeuralNetworkMTLR
from torchmtlr import mtlr_survival
from torchmtlr.utils import encode_survival
from torch.utils.data import SequentialSampler, DataLoader
from transformations import RangeScaler
from utilities import maybe_create_directories


def maybe_expand_dims(x: Tensor):
    if x.dim() == 1:
        x = x.reshape([1, -1])
    return x


def load_dataset(
        root: str,
        files: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None
) -> CancerDataset:
    """
    Loads a dataset from serialized data on disk.

    :param root: The path to the directory where the data are stored. Should contain a "raw" subdirectory containing
    the serialized data files.
    :param files: A collection of files to use from the 'raw' subdirectory of `root`
    :param transform: A transformation to apply to the loaded data
    :return: A CancerDataset
    """

    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    data_files = sorted(os.listdir(os.path.join(root, "raw")))  # Raw graph Data object files
    if files is not None:
        data_files = [file for file in data_files if file in files]
    # transformation = StandardizeFeatures(correction=1)  # No transform
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerDataset(root=root, data_files=data_files, transform=transform)
    return dataset


def get_latent_representation(input_tensor, module):
    # https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model
    with torch.no_grad():
        nn_module_output = None

        def nn_block_hook(module_, input_, output_):
            nonlocal nn_module_output
            nn_module_output = output_

        hook = module.nn_module.register_forward_hook(nn_block_hook)
        module(input_tensor)
        hook.remove()

        return nn_module_output


def main():

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Export results from cross-validated MLP MTLR survival models"
    )
    parser.add_argument(
        "-d", "--data_dir",
        help="The path of the directory where data necessary for training the models are located",
        type=str
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path of the directory where experimental results will be written",
        type=str
    )
    parser.add_argument(
        "-t", "--cancer_type",
        help="SCANB or TCGA letter code of cancer type for which to train models",
        type=str
    )
    parser.add_argument(
        "-db", "--database",
        help="The database the graphs are constructed from. Takes \"reactome\" or \"kegg\".",
    )
    parser.add_argument(
        "--directed",
        help="If using reactome graphs, use this flag to learn over directed graphs. The default behavior is to use "
             "reactome graphs with both directed and undirected edges.",
        action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size",
        help="The mini-batch size for training and testing",
        type=int
    )
    parser.add_argument(
        "--sagpool",
        help="If set, the script will use SAGPool layers in the GNN model",
        dest="sagpool", action="store_true"
    )
    parser.add_argument(
        "--use_clin_feats",
        help="Use age at initial pathologic diagnosis and pathologic stage as auxiliary input features to the MLP "
             "block that outputs the survival predictions",
        dest="use_clin_feats", action="store_true"
    )
    parser.add_argument(
        "--normalize_gene_exprs",
        help="Normalize gene expression values within each biopsy to a 0-1 range by subtracting from each value the "
             "within-biopsy minimum and dividing by the within-biopsy range",
        dest="normalize_gene_exprs", action="store_true"
    )
    args = vars(parser.parse_args())
    # endregion Parse args

    # region Define important values
    data_dir = args["data_dir"]  # e.g. ./data
    output_dir = args["output_dir"]  # e.g. ./experiment5
    cancer_type = args["cancer_type"]
    model_type = "mlp"
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
    use_sagpool = args["sagpool"]
    sagpool = "sagpool" if use_sagpool else "no-sagpool"
    batch_size = args["batch_size"]
    use_clin_feats = True if args["use_clin_feats"] else False
    normalize_gene_exprs = True if args["normalize_gene_exprs"] else False
    # endregion Define important values

    # region Directories
    cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer_type))
    cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer_type))
    input_data_dir = os.path.join(cancer_data_dir, "mlp_inputs", database, direction)  # Directory with inputs to MLP

    shared_path_sgmt = os.path.join(model_type, database, direction)
    model_dir = os.path.join(cancer_out_dir, "models", shared_path_sgmt)
    hp_dir = os.path.join(cancer_out_dir, "hyperparameters", shared_path_sgmt)
    export_dir = os.path.join(cancer_out_dir, "exports", shared_path_sgmt)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(export_dir)
    # endregion Directories

    # region Files to read
    file_name = f"{cancer_type}_{database}_{direction}_mlp_mask.pt"
    mask_matrix_file = os.path.join(data_dir, cancer_type, file_name)
    hp_file = os.path.join(hp_dir, f"{cancer_type}_{model_type}_{database}_{direction}_hyperparameters.pkl")
    # endregion Files to read

    # region Load data
    # Load tuned MLP hyperparameters
    with open(hp_file, "rb") as file_in:
        hp_dict = pickle.load(file_in)

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(cancer_data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_test_names = pickle.load(file_in)
    n_folds = len(train_test_names)
    start_fold = 1  # Folds 1 through k - 1 are used for model evaluation
    # endregion Load data

    transform = RangeScaler(dim=0) if normalize_gene_exprs else None

    shared_prefix = f"{cancer_type}_{model_type}_{database}_{direction}"

    for k in range(start_fold, n_folds):

        # Load weight mask for SparseMLP Module
        with open(mask_matrix_file, "rb") as file_in:
            buffer = io.BytesIO(file_in.read())
        pathway_mask = torch.load(buffer)

        # Model file name
        model_file = os.path.join(model_dir, f"{shared_prefix}_model_feature-selection={sagpool}_fold={k}.pt")

        # Get the names and raw Data file list indices of the biopsies in the current CV fold
        train_names, val_names = train_test_names[k]
        # Create training, validation datasets
        train_dataset = load_dataset(input_data_dir, train_names, transform=transform)
        val_dataset = load_dataset(input_data_dir, val_names)

        # Initialize dataloaders
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)

        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)

        # Load saved intervals, input feature indices to use, and model weights for the current fold
        saved_model_objects = torch.load(model_file)
        intervals, feature_indices, sparse_layer_state, model_state, _ = saved_model_objects

        # Edit the sparse layer state dict keys
        new_sparse_layer_state = OrderedDict()
        for key, val in sparse_layer_state.items():
            key = re.sub(r"\bmodule\.", "", key)
            new_sparse_layer_state[key] = val
        sparse_layer_state = new_sparse_layer_state
        del new_sparse_layer_state

        # Edit the neural network MTLR block state dict keys
        new_model_state = OrderedDict()
        for key, val in model_state.items():
            key = re.sub(r"\bmodule\.", "", key)
            new_model_state[key] = val
        model_state = new_model_state
        del new_model_state

        # Restrict the mask to input genes (columns) that will be used by the sparse MLP
        if feature_indices is not None:
            pathway_mask = pathway_mask[:, feature_indices]

        if use_clin_feats:
            n_aux_feats = len(train_dataset[0][0]) - 1
            total_feats = pathway_mask.shape[0] + n_aux_feats
        else:
            total_feats = pathway_mask.shape[0]

        # Initialize models
        sparse_layer = SparseMLP(pathway_mask)
        sparse_layer.load_state_dict(sparse_layer_state)

        model = NeuralNetworkMTLR(num_time_bins=len(intervals), in_features=total_feats)
        model.load_state_dict(model_state)

        # Forward passes
        # A forward pass through the model returns [batch_size, n_intervals] logits that express the log-probability
        # that the event occurred in each interval
        sparse_layer.eval()
        model.eval()
        with torch.no_grad():

            # Training data
            train_pathway_scores = []
            train_latent_repr = []
            train_predictions = []
            train_times = []
            train_censor_bits = []
            train_surv_seqs = []  # Binary sequences encoding survival times

            # Test data
            val_pathway_scores = []
            val_latent_repr = []
            val_predictions = []
            val_times = []
            val_censor_bits = []
            val_surv_seqs = []  # Binary sequences encoding survival times

            for (exprs, age, stage), (surv_times, censor_bits) in train_dataloader:

                current_batch_size = exprs.shape[0]
                exprs = torch.reshape(exprs, (current_batch_size, -1))
                if feature_indices is not None:
                    exprs = exprs[:, feature_indices]

                pathway_scores = sparse_layer(exprs)
                age = age.reshape([-1, 1])
                stage = stage.reshape([-1, 1])
                inputs = torch.cat([pathway_scores, age, stage], dim=-1) if use_clin_feats else pathway_scores

                latent_representation = get_latent_representation(inputs, model)
                predictions = model(inputs)

                train_pathway_scores.append(pathway_scores)
                train_latent_repr.append(latent_representation)  # From forward hook
                train_predictions.append(maybe_expand_dims(mtlr_survival(predictions)))
                train_times.append(surv_times)
                train_censor_bits.append(censor_bits)
                train_surv_seqs.append(maybe_expand_dims(encode_survival(surv_times, censor_bits, intervals)))

            for (exprs, age, stage), (surv_times, censor_bits) in val_dataloader:

                current_batch_size = exprs.shape[0]
                exprs = torch.reshape(exprs, (current_batch_size, -1))
                if feature_indices is not None:
                    exprs = exprs[:, feature_indices]

                pathway_scores = sparse_layer(exprs)
                age = age.reshape([-1, 1])
                stage = stage.reshape([-1, 1])
                inputs = torch.cat([pathway_scores, age, stage], dim=-1) if use_clin_feats else pathway_scores

                latent_representation = get_latent_representation(inputs, model)
                predictions = model(inputs)

                val_pathway_scores.append(pathway_scores)
                val_latent_repr.append(latent_representation)  # From forward hook
                val_predictions.append(maybe_expand_dims(mtlr_survival(predictions)))
                val_times.append(surv_times)
                val_censor_bits.append(censor_bits)
                val_surv_seqs.append(maybe_expand_dims(encode_survival(surv_times, censor_bits, intervals)))

        # region Export data
        train_predictions = torch.cat(train_predictions, dim=0).numpy()
        val_predictions = torch.cat(val_predictions, dim=0).numpy()

        bias_tensor = model.mtlr.mtlr_bias.data.unsqueeze(1)
        weight_tensor = model.mtlr.mtlr_weight.data.transpose(0, 1)
        weight_tensor = torch.cat(tensors=[bias_tensor, weight_tensor], dim=1)
        weight_array = weight_tensor.numpy()

        x_train_array = torch.cat(train_latent_repr, dim=0).numpy()
        x_val_array = torch.cat(val_latent_repr, dim=0).numpy()

        time_train_array = torch.cat(train_times, dim=0).numpy()
        time_val_array = torch.cat(val_times, dim=0).numpy()

        cns_train_array = torch.cat(train_censor_bits, dim=0).numpy()
        cns_val_array = torch.cat(val_censor_bits, dim=0).numpy()

        seq_train_array = torch.cat(train_surv_seqs, dim=0).numpy()
        seq_val_array = torch.cat(val_surv_seqs, dim=0).numpy()

        time_points_array = intervals.numpy()

        # Write data to csv files
        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_test_predictions_{k}.csv")
        np.savetxt(file_path, val_predictions, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_mtlr_weights_{k}.csv")
        np.savetxt(file_path, weight_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_latent_train_{k}.csv")
        np.savetxt(file_path, x_train_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_latent_test_{k}.csv")
        np.savetxt(file_path, x_val_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_seq_train_{k}.csv")
        np.savetxt(file_path, seq_train_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_seq_test_{k}.csv")
        np.savetxt(file_path, seq_val_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_time_train_{k}.csv")
        np.savetxt(file_path, time_train_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_time_test_{k}.csv")
        np.savetxt(file_path, time_val_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_cns_train_{k}.csv")
        np.savetxt(file_path, cns_train_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_cns_test_{k}.csv")
        np.savetxt(file_path, cns_val_array, delimiter=",")

        file_path = os.path.join(export_dir, f"{shared_prefix}_{sagpool}_time_points_{k}.csv")
        np.savetxt(file_path, time_points_array, delimiter=",")
        # endregion Export data

    c1_array = np.array([hp_dict["C1"]])
    np.savetxt(os.path.join(export_dir, f"{shared_prefix}"), c1_array, delimiter=",")


if __name__ == "__main__":
    main()
