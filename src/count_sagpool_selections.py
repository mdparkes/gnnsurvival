import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import seaborn as sns
import torch

from collections import OrderedDict
from filelock import FileLock
from torch import Tensor
from torch.utils.data import SequentialSampler
from torch_geometric.loader import DataLoader
from typing import Callable, Optional

from dataset_definitions import CancerGraphDataset
from models import NeuralNetworkMTLR, IndividualPathsMPNN


def backtrace_sagpool_selections(*node_indices, n_original_nodes: int, batch_size: int, ptr: Tensor) -> Tensor:
    """
    Backtraces through a series of indices of unmasked node indices to identify the original indices of the
    nodes that survived through all the SAGPool layers

    :param node_indices: Tensors of indices of unmasked nodes returned as "perm" by each SAGPool layer. The indices
    should be supplied in order of increasing layer depth in the model. In other words, the indices returned by the
    earliest SAGPool layer should be first, and the indices returned by the final SAGPool layer should be last.
    :param n_original_nodes: The original number of nodes in the input graph. If the input to the GNN was a batch
    of graphs, this should be the total number of nodes across all graphs in the batch: batch_size * n_graph_nodes.
    :param batch_size: The number of graphs in the input batch
    :param ptr: The indices of the first nodes of each batch in the input tensor. The size should be batch_size + 1,
    and the final element of ptr should be n_original_nodes.
    :return: A Tensor of unmasked node indices with respect to the original input graph's nodes, starting from zero,
    which can be mapped to human-readable gene IDs.
    """
    original_indices = torch.arange(n_original_nodes, requires_grad=False)
    for indices in node_indices:
        original_indices = original_indices[indices]
    # The pathway graphs in each DataBatch are structurally identical, but since the node features differ
    # the SAGPool layer may return different node selections for each biopsy in the batch.
    pooled_pathway_size = original_indices.size(0) // batch_size  # Assumes all graphs have the same number of nodes
    offset = torch.repeat_interleave(ptr[:-1], pooled_pathway_size)
    # Get the indices of nodes that were retained from the original graph for each input in the batch
    original_indices = original_indices - offset  # Indices of retained nodes
    original_indices = torch.reshape(original_indices, (batch_size, -1))  # Rows are graphs in the batch
    original_indices, _ = torch.sort(original_indices, 1)  # Retained node indices sorted in ascending order
    return original_indices


def gnn_forward_pass(data_batch_list, mp_modules, model, relational, aux_features=None):
    """
    Perform a forward pass through the GNN modules and model.

    :param data_batch_list: A list of DataBatch or HeteroDataBatch objects, one per input graph, each representing a
    batch of biopsies.
    :param mp_modules: A ModuleList with the message passing module for each input graph
    :param model: The neural network that takes a vector of graph scores as input and returns a survival distribution
    :param relational: Set this to True if input graphs are relational/heterogeneous
    :param aux_features: A tensor of auxiliary features to use as input. If supplied, these will be concatenated with
    the results of forward passes of graphs through mp_modules. The concatenation is used as input to the final MLP
    block that outputs predictions.
    :return: Predictions for a batch of biopsies and a list of tuples of indices of nodes retained by each SAGPool
    layer for each input graph with a batch of biopsies.
    """
    pathway_scores = list()  # Populated with a list of [n_pathways] shape [batch_size, 1] tensors of pathway scores
    nodes_retained_list = list()
    if relational:
        for i, graph in enumerate(data_batch_list):
            score, _, nodes_retained = mp_modules[i](graph.x_dict, graph.edge_index_dict, graph.batch_dict)
            pathway_scores.append(score)
            nodes_retained_list.append(nodes_retained)
    else:
        for i, graph in enumerate(data_batch_list):
            score, _, nodes_retained = mp_modules[i](graph.x, graph.edge_index, graph.batch)
            pathway_scores.append(score)
            nodes_retained_list.append(nodes_retained)
    inputs = torch.cat(pathway_scores, dim=-1)  # shape [batch_size, n_pathways]
    if aux_features is not None:
        current_batch_size = inputs.shape[0]
        aux_features = torch.cat(aux_features, dim=-1).reshape([current_batch_size, -1])
        inputs = torch.cat([inputs, aux_features], dim=-1)  # shape [batch_size, n_pathways + n_aux_features]
    predictions = model(inputs)
    return predictions, nodes_retained_list


def save_sagpool_selection_frequencies(
        model_f, feat_names_f, norm_freq_f, unnorm_freq_f, sagpool_ratio, batch_sz, dataset, relational, use_aux_feats
) -> None:
    # region Load best model
    time_intervals, mp_mods_state, mod_state, _ = torch.load(model_f)
    new_mp_mods_state = OrderedDict()
    for key, val in mp_mods_state.items():
        key = re.sub(r"\bmodule\.", "", key)
        new_mp_mods_state[key] = val
    new_mod_state = OrderedDict()
    for key, val in mod_state.items():
        key = re.sub(r"\bmodule\.", "", key)
        new_mod_state[key] = val

    mp_modules = torch.nn.ModuleList()

    n_submodules = len(dataset[0][0][0])  # Number of different pathways
    if use_aux_feats:
        n_aux_feats = len(dataset[0][0]) - 1  # Number of auxiliary features
        total_feats = n_submodules + n_aux_feats
    else:
        total_feats = n_submodules

    for i in range(n_submodules):  # Initialize modules for non-relational graphs
        num_nodes = int(dataset[0][0][0][i].x.size(0))  # Number of nodes in the pathway graph
        mp_mod = IndividualPathsMPNN(message_passing="graphsage", use_sagpool=True, ratio=sagpool_ratio,
                                     num_nodes=num_nodes)
        mp_modules.append(mp_mod)
    mp_modules.load_state_dict(new_mp_mods_state)

    model = NeuralNetworkMTLR(num_time_bins=len(time_intervals), in_features=total_feats)
    model.load_state_dict(new_mod_state)
    # endregion Load best model

    # Load the names of the genes represented by each node in the graphs
    with open(feat_names_f, "rb") as file_in:
        feature_names = pickle.load(file_in)
    # Initialize a dict of the number of times each gene survived SAGPool
    unique_gene_names = set()
    for pathway_gene_list in feature_names:
        unique_gene_names = unique_gene_names.union(set(pathway_gene_list))
    gene_retention_freq = {k: 0 for k in unique_gene_names}
    """
    If pathway graphs are being fed through the model one at a time, SAGPool operates within pathways
    and guarantees that at least one node will survive pooling in each pathway. Some genes are involved
    in multiple pathways and have more opportunities to be selected by SAGPool. Therefore, the number
    of times they survive pooling may be a consequence of the number of pathway graphs they appear in,
    and not necessarily a consequence of their relevance to the prediction task.

    Conversely, when all pathways graphs are passed through the model at once as a large conglomerate
    graph, SAGPooling occurs across pathways and presumably only emphasizes nodes that are important to the
    prediction task. There is no guarantee that even a single node in a given pathway will survive SAGPooling when 
    all pathway graphs are merged. In this case we take a gene's retention by SAGPool at face value and do not 
    normalize the counts by the number of pathways the gene participates in.
    """
    pathways_appeared_in = {k: 0 for k in unique_gene_names}  # Number of times a gene appears in a pathway
    for gene in unique_gene_names:
        for pathway_gene_list in feature_names:
            if gene in pathway_gene_list:
                pathways_appeared_in[gene] += 1

    # Sample batch_size patients from the training set in sequential order for each graph
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_sz, drop_last=False)

    mp_modules.eval()  # Set message passing modules to evaluation mode
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for data in dataloader:
            bs = len(data[0][0][0])  # Actual batch size. The final batch may have fewer than batch_size observations
            (data_batch_list, age, stage), _ = data
            aux_features = [age, stage] if use_aux_feats else None
            _, nodes_retained = gnn_forward_pass(data_batch_list, mp_modules, model, relational, aux_features)
            # nodes_retained is a list of tuples of batched node indices retained by each SAGPool layer
            for i, perm_tuple in enumerate(nodes_retained):
                num_nodes = len(feature_names[i])  # Original number of nodes in the unpooled graph
                ptr = torch.tensor([num_nodes * j for j in range(bs + 1)])  # Need that + 1.
                # Backtrace selections to the corresponding indices in the original unpooled graph
                idx_tns = torch.flatten(backtrace_sagpool_selections(
                    *perm_tuple, n_original_nodes=num_nodes * bs,
                    batch_size=bs, ptr=ptr
                ))
                # Get the names of genes retained by SAGPool in the graph batch and update their counts
                genes_retained = [feature_names[i][j] for j in idx_tns]
                for gene in genes_retained:
                    gene_retention_freq[gene] += 1
    # Normalize the gene frequencies by the number of pathways each gene appears in, sort, and save
    normalized_freq = [c / z for c, z in zip(gene_retention_freq.values(), pathways_appeared_in.values())]
    normalized_gene_retention_freq = dict(zip(gene_retention_freq.keys(), normalized_freq))
    sorted_counts = sorted(normalized_gene_retention_freq.items(), reverse=True, key=lambda item: item[1])
    normalized_gene_retention_freq = {k: v for k, v in sorted_counts}
    with open(norm_freq_f, "wb") as file_out:
        pickle.dump(normalized_gene_retention_freq, file_out)
    # Sort the unnormalized gene frequency dict in descending order of count and save
    sorted_counts = sorted(gene_retention_freq.items(), reverse=True, key=lambda item: item[1])
    gene_retention_freq = {k: v for k, v in sorted_counts}
    with open(unnorm_freq_f, "wb") as file_out:
        pickle.dump(gene_retention_freq, file_out)


def load_dataset(graph_dir: str, transform: Optional[Callable] = None) -> CancerGraphDataset:
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    graph_files = sorted(os.listdir(os.path.join(graph_dir, "raw")))  # Raw graph Data object files
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerGraphDataset(root=graph_dir, data_files=graph_files, transform=transform)
    return dataset


def main():
    raise NotImplementedError


if __name__=="__main__":

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Count the number of times each gene is retained by SAGPooling after all forward passes of "
                    "a dataset"
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
        help="TCGA letter code of cancer type for which to train models",
        type=str
    )
    parser.add_argument(
        "-db", "--database",
        help="The database the graphs are constructed from. Takes \"reactome\" or \"kegg\".",
    )
    parser.add_argument(
        "--directed",
        help="If using reactome graphs, use this flag for directed graphs. The default behavior is to use "
             "reactome graphs with both directed and undirected edges. No effect if using BRITE graphs.",
        action="store_true"
    )
    parser.add_argument(
        "--relational",
        help="Use relational GNN with one edge set per gene interaction type. Relational graphs use all "
             "interaction types except \"relation,\" which signifies any type of interaction.",
        action="store_true"
    )
    parser.add_argument(
        "--nonrelational",
        help="Use a non-relational GNN with one edge set.",
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
    graph_type = "relational" if args["relational"] else "nonrelational"
    relational = True if graph_type == "relational" else False
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
    merge_pathways = True if args["merge_pathways"] else False
    merge = "merged" if merge_pathways else "unmerged"
    use_sagpool = True
    sagpool = "sagpool"
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
    export_dir = os.path.join(cancer_out_dir, "exports", model_type, shared_path_sgmt)
    hp_dir = os.path.join(cancer_out_dir, "hyperparameters", model_type, shared_path_sgmt)
    # endregion Directories

    # region Files to read/write
    graph_data_files = sorted(os.listdir(os.path.join(graph_data_dir, "raw")))  # Raw graph Data object files
    # HSA/ENTREZ IDs of features (genes) in the GNN input graph
    file_name = f"{cancer_type}_{database}_{merge}_{direction}_feature_names.pkl"
    feature_names_file = os.path.join(cancer_data_dir, file_name)

    shared_prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}"
    hp_file = os.path.join(hp_dir, f"{shared_prefix}_{sagpool}_hyperparameters.pkl")
    # endregion Files to read/write

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(cancer_data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_test_names = pickle.load(file_in)
    n_folds = len(train_test_names)
    start_fold = 1  # Folds 1 through k - 1 are used for model evaluation
    tuning_fold = 0  # Fold 0 is reserved exclusively for hyperparameter tuning
    # Load hyperparameters
    with open(hp_file, "rb") as file_in:
        hp_dict = pickle.load(file_in)
    # Load dataset
    ds = load_dataset(graph_data_dir)

    for k in range(start_fold, n_folds):
        # Number of times each gene was retained after all observations were passed through the GNN with sagpool
        file_name = f"unnormalized_sagpool_gene_retention_frequencies_{k}.pkl"
        unnormalized_freq_file = os.path.join(cancer_out_dir, "models", "gnn", shared_path_sgmt, file_name)
        # Gene retention frequencies normalized by the number of times each gene appears in a pathway
        file_name = f"normalized_sagpool_gene_retention_frequencies_{k}.pkl"
        normalized_freq_file = os.path.join(cancer_out_dir, "models", "gnn", shared_path_sgmt, file_name)
        # Path to file containing model state dicts
        model_file = os.path.join(model_dir, f"{shared_prefix}_model_feature-selection={sagpool}_fold={k}.pt")
        # Get the names of the raw Data files for biopsies the current CV fold
        train_names, val_names = train_test_names[tuning_fold]
        # Get the indices of the current CV fold's raw Data files in the file list
        train_idx = [graph_data_files.index(name) for name in train_names]
        val_idx = [graph_data_files.index(name) for name in val_names]
        # region Count the number of times each gene was retained by SAGPool in the GNN
        save_sagpool_selection_frequencies(
            model_f=model_file,
            feat_names_f=feature_names_file,
            norm_freq_f=normalized_freq_file,
            unnorm_freq_f=unnormalized_freq_file,
            sagpool_ratio=hp_dict["ratio"],
            batch_sz=batch_size,
            dataset=ds[train_idx],
            relational=relational,
            use_aux_feats=use_clin_feats
        )

        # Load gene frequencies and plot a histogram of counts
        with open(unnormalized_freq_file, "rb") as file_in:
            unnormalized_counts = pd.Series(pickle.load(file_in))

        with open(normalized_freq_file, "rb") as file_in:
            normalized_counts = pd.Series(pickle.load(file_in))

        # Plot normalized gene counts
        plt_df = pd.DataFrame({"Gene": np.arange(len(normalized_counts)), "Count": normalized_counts.values})
        fig, ax = plt.subplots(figsize=(8, 8))
        p1 = sns.lineplot(x="Gene", y="Count", data=plt_df)
        p1.set(xticklabels=[], title="Normalized SAGPool gene selection counts", ylabel="Count", xlabel="Gene")
        p1.tick_params(bottom=False)
        plt.savefig(os.path.join(export_dir, "normalized_sagpool_counts.png"))

        # Plot unnormalized gene counts
        plt_df = pd.DataFrame({"Gene": np.arange(len(unnormalized_counts)), "Count": unnormalized_counts.values})
        fig, ax = plt.subplots(figsize=(8, 8))
        p2 = sns.lineplot(x="Gene", y="Count", data=plt_df)
        p2.set(xticklabels=[], title="SAGPool gene selection counts", ylabel="Count", xlabel="Gene")
        p2.tick_params(bottom=False)
        plt.savefig(os.path.join(export_dir, "unnormalized_sagpool_counts.png"))
