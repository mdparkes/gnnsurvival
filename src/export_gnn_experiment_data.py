import argparse
import io
import numpy as np
import os
import pickle
import re
import torch

from collections import OrderedDict
from filelock import FileLock
from torchmtlr import mtlr_survival
from torchmtlr.utils import encode_survival
from torch_geometric.loader import DataLoader
from torch.utils.data import SequentialSampler
from typing import Callable, Optional


from dataset_definitions import CancerGraphDataset
from models import IndividualPathsMPNN, NeuralNetworkMTLR
from transformations import GraphRangeScaler
from utilities import maybe_create_directories


def maybe_expand_dims(x: torch.Tensor):
    if x.dim() == 1:
        x = x.reshape([1, -1])
    return x


def load_dataset(graph_dir: str, transform: Optional[Callable] = None) -> CancerGraphDataset:
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    graph_files = sorted(os.listdir(os.path.join(graph_dir, "raw")))  # Raw graph Data object files
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerGraphDataset(root=graph_dir, data_files=graph_files, transform=transform)
    return dataset


def mp_module_forward_pass(data_batch_list, mp_modules, use_relational):
    """
    Perform a forward pass through the GNN message passing modules. Returns a Tensor of pathway scores and a list of
    graph node indices that were retained after each SAGPool operation.

    :param data_batch_list: A list of DataBatch or HeteroDataBatch objects, one per input graph, each representing a
    batch of biopsies.
    :param mp_modules: A ModuleList with the message passing module for each input graph
    :param use_relational: Set this to True if input graphs are relational/heterogeneous
    :return: A pathway scores Tensor for a batch of biopsies and a list of tuples of indices of nodes retained by each
    SAGPool layer for each input graph with a batch of biopsies.
    """
    pw_score_list = list()  # Populated with a list of [n_pathways] shape [batch_size, 1] tensors of pathway scores
    nodes_retained_list = list()
    if use_relational:
        for i, graph in enumerate(data_batch_list):
            score, _, nodes_retained = mp_modules[i](graph.x_dict, graph.edge_index_dict, graph.batch_dict)
            pw_score_list.append(score)
            nodes_retained_list.append(nodes_retained)
    else:
        for i, graph in enumerate(data_batch_list):
            score, _, nodes_retained = mp_modules[i](graph.x, graph.edge_index, graph.batch)
            pw_score_list.append(score)
            nodes_retained_list.append(nodes_retained)
    pw_scores = torch.cat(pw_score_list, dim=-1)  # shape [batch_size, n_pathways]
    return pw_scores, nodes_retained_list


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
        description="Export results from cross-validated GNN MTLR survival models"
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
             "reactome graphs with both directed and undirected edges. No effect if using BRITE graphs.",
        action="store_true"
    )
    parser.add_argument(
        "--relational",
        help="Train a relational GNN with one edge set per gene interaction type. Relational graphs use all "
             "interaction types except \"relation,\" which signifies any type of interaction.",
        action="store_true"
    )
    parser.add_argument(
        "--nonrelational",
        help="Train a non-relational GNN with one edge set.",
        dest="relational",
        action="store_false"
    )
    parser.add_argument(
        "--merge_pathways",
        help="If set, the script will merge all pathways into a single large graph per patient. If not set, "
             "the default behavior is to create separate pathway graphs that will be fed through neural networks "
             "individually or in batches per patient.",
        dest="merge_pathways", action="store_true"
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
    output_dir = args["output_dir"]  # e.g. ./experiment6
    cancer_type = args["cancer_type"]
    model_type = "gnn"
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    merge_pathways = True if args["merge_pathways"] else False
    merge = "merged" if merge_pathways else "unmerged"
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
    graph_type = "relational" if args["relational"] else "nonrelational"
    relational = True if graph_type == "relational" else False
    use_sagpool = args["sagpool"]
    sagpool = "sagpool" if use_sagpool else "no-sagpool"
    batch_size = args["batch_size"]
    use_clin_feats = True if args["use_clin_feats"] else False
    normalize_gene_exprs = True if args["normalize_gene_exprs"] else False
    # endregion Define important values

    # region Directories
    shared_path_sgmt = os.path.join(database, merge, direction, graph_type)
    cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer_type))
    cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer_type))
    graph_data_dir = os.path.join(cancer_data_dir, "graphs", shared_path_sgmt)
    model_dir = os.path.join(cancer_out_dir, "models", model_type, shared_path_sgmt)
    hp_dir = os.path.join(cancer_out_dir, "hyperparameters", model_type, shared_path_sgmt)
    export_dir = os.path.join(cancer_out_dir, "exports", model_type, shared_path_sgmt)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(export_dir)
    # endregion Directories

    # region Files to read
    graph_data_files = sorted(os.listdir(os.path.join(graph_data_dir, "raw")))  # Raw graph Data object files
    # Gene to pathway graph coarsening scheme for pooling graphs if merge_pathways is True
    file_name = f"{cancer_type}_{database}_{direction}_pooling_assignment.pt"  # Only used if merge_pathways
    assignment_matrix_file = os.path.join(data_dir, cancer_type, file_name) if merge_pathways else None
    # Hyperparameters
    shared_prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}_{sagpool}"
    hp_file = os.path.join(hp_dir, f"{shared_prefix}_hyperparameters.pkl")
    # endregion Files to read

    # region Load data
    if assignment_matrix_file is not None:  # Only used if merge_pathways
        with open(assignment_matrix_file, "rb") as file_in:
            buffer = io.BytesIO(file_in.read())
        pooling_assignment = torch.load(buffer)

    # Load tuned GNN hyperparameters
    with open(hp_file, "rb") as file_in:
        hp_dict = pickle.load(file_in)

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(cancer_data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_test_names = pickle.load(file_in)
    n_folds = len(train_test_names)
    start_fold = 1  # Folds 1 through k - 1 are used for model evaluation

    # Load dataset
    transform = GraphRangeScaler(attrs=["x"], dim=0) if normalize_gene_exprs else None
    ds = load_dataset(graph_data_dir, transform=transform)
    # endregion Load data

    shared_prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}"

    for k in range(start_fold, n_folds):

        # Model file name
        model_file = os.path.join(model_dir, f"{shared_prefix}_model_feature-selection={sagpool}_fold={k}.pt")

        # Get the names and raw Data file list indices of the biopsies in the current CV fold
        train_names, val_names = train_test_names[k]
        train_idx = [graph_data_files.index(name) for name in train_names]
        val_idx = [graph_data_files.index(name) for name in val_names]
        # Create training, validation datasets
        train_dataset = ds[train_idx]
        val_dataset = ds[val_idx]

        # Initialize dataloaders
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)

        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)

        # Load saved intervals, input feature indices to use, and model weights for the current fold
        saved_model_objects = torch.load(model_file)
        intervals, mp_modules_state, model_state, _ = saved_model_objects

        # Edit the sparse layer state dict keys
        new_mp_modules_state = OrderedDict()
        for key, val in mp_modules_state.items():
            key = re.sub(r"\bmodule\.", "", key)
            new_mp_modules_state[key] = val
        mp_modules_state = new_mp_modules_state
        del new_mp_modules_state

        # Edit the neural network MTLR block state dict keys
        new_model_state = OrderedDict()
        for key, val in model_state.items():
            key = re.sub(r"\bmodule\.", "", key)
            new_model_state[key] = val
        model_state = new_model_state
        del new_model_state

        # Initialize models
        mp_modules = torch.nn.ModuleList()

        n_submodules = len(ds[0][0][0])  # Number of different pathways
        if use_clin_feats:
            n_aux_feats = len(ds[0][0]) - 1  # Number of auxiliary features
            total_feats = n_submodules + n_aux_feats
        else:
            total_feats = n_submodules

        for i in range(n_submodules):
            num_nodes = int(ds[0][0][0][i].x.size(0))
            mp_module = IndividualPathsMPNN(message_passing="graphsage", use_sagpool=use_sagpool,
                                            ratio=hp_dict["ratio"], num_nodes=num_nodes)
            mp_modules.append(mp_module)
        mp_modules.load_state_dict(mp_modules_state)

        model = NeuralNetworkMTLR(num_time_bins=len(intervals), in_features=total_feats)
        model.load_state_dict(model_state)

        # Forward passes
        # A forward pass through the model returns [batch_size, n_intervals] logits that express the log-probability
        # that the event occurred in each interval
        mp_modules.eval()
        model.eval()
        with torch.no_grad():
            # Training data
            train_pathway_scores = []
            train_latent_repr = []
            train_predictions = []
            train_times = []
            train_censor_bits = []
            train_surv_seqs = []  # Binary sequences encoding survival times
            for (exprs, age, stage), (surv_times, censor_bits) in train_dataloader:
                pathway_scores, _ = mp_module_forward_pass(exprs, mp_modules, relational)
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
            # Test data
            val_pathway_scores = []
            val_latent_repr = []
            val_predictions = []
            val_times = []
            val_censor_bits = []
            val_surv_seqs = []  # Binary sequences encoding survival times
            for (exprs, age, stage), (surv_times, censor_bits) in val_dataloader:
                pathway_scores, _ = mp_module_forward_pass(exprs, mp_modules, relational)
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

