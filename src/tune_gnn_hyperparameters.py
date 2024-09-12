"""
Perform hyperparameter tuning using a fold of data that is exclusively reserved for this purpose.
"""

import argparse
import io
import os
import pickle
import tempfile

import torch
import torch_geometric.typing

from filelock import FileLock
from matplotlib.ticker import MaxNLocator
from ray import tune, train
from ray.train import Checkpoint
from torch import Tensor
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torchmtlr import mtlr_neg_log_likelihood
from torchmtlr.utils import encode_survival, make_time_bins
from typing import Any, Callable, Dict, Optional, Sequence

from dataset_definitions import CancerGraphDataset
from models import NeuralNetworkMTLR, IndividualPathsMPNN
from transformations import GraphRangeScaler
from utilities import maybe_create_directories


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


def train_loop(
        config: Dict[str, Any],
        *,
        dataset: CancerGraphDataset,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        batch_size: int,
        use_relational_graphs: bool,
        use_sagpool: bool,
        time_intervals: Tensor,
        use_aux_feats: bool
) -> None:

    epochs = 150

    # region Create dataloaders
    # Create samplers that partition stratified CV folds into disjoint random batches
    train_batch_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size, drop_last=False)
    val_batch_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size, drop_last=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_batch_sampler)
    val_dataloader = DataLoader(dataset, batch_sampler=val_batch_sampler)
    # endregion Create dataloaders

    # region Initialize models and optimizer
    n_submodules = len(dataset[0][0][0])  # Number of different pathways
    if use_aux_feats:
        n_aux_feats = len(dataset[0][0]) - 1  # Number of auxiliary features
        total_feats = n_submodules + n_aux_feats
    else:
        total_feats = n_submodules

    mp_modules = torch.nn.ModuleList()
    if use_relational_graphs:
        raise NotImplementedError("Relational GNN models not implemented")
    else:
        for i in range(n_submodules):
            num_nodes = int(dataset[0][0][0][i].x.size(0))  # Number of nodes in the pathway graph
            mp_mod = IndividualPathsMPNN(message_passing="graphsage", use_sagpool=use_sagpool,
                                         ratio=config["ratio"], num_nodes=num_nodes)
            mp_modules.append(mp_mod)

    model = NeuralNetworkMTLR(num_time_bins=len(time_intervals), in_features=total_feats)

    optimizer = torch.optim.Adam([
        {"params": mp_modules.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch, mp_modules_state, model_state, optimizer_state = ckpt
            mp_modules.load_state_dict(mp_modules_state)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    else:
        start_epoch = 0

    train_losses = []  # Append epoch losses
    val_losses = []  # Append epoch losses

    for t in range(start_epoch, epochs):

        # region Train
        mp_modules.train()
        model.train()
        epoch_train_loss = 0.
        samples_processed = 0
        for i, loaded_data in enumerate(train_dataloader):

            (data_batch_list, age, stage), label_tensor_list = loaded_data
            current_batch_size = len(data_batch_list[0])
            samples_processed += current_batch_size

            # data_batch_list is a list of m DataBatch objects, where m is the number of graphs fed through the GNN
            # for a single patient. Each DataBatch object represents a batch of a particular graph.
            # label_tensor_list is a length-2 list. The first item is a shape [n] Tensor of survival times (where n
            # is the number of biopsies in the batch). The second item is a shape [n] Tensor of censor bits.
            targets = encode_survival(*label_tensor_list, bins=time_intervals)
            targets = torch.reshape(targets, (current_batch_size, -1))

            if use_aux_feats:
                predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs,
                                                  aux_features=[age, stage])
            else:
                predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs,
                                                  aux_features=None)
            predictions = torch.reshape(predictions, (current_batch_size, -1))
            # Calculate training loss for the batch
            loss = mtlr_neg_log_likelihood(predictions, targets, model, C1=config["C1"], average=False)
            epoch_train_loss += loss  # Running total of this epoch's training loss
            loss /= current_batch_size  # Per-patient loss for the current batch
            # Optimize model weights w.r.t. per-patient loss for the current batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= samples_processed  # Per-patient average training loss for this epoch
        train_losses.append(float(epoch_train_loss))
        # endregion Train

        # region Evaluate
        mp_modules.eval()
        model.eval()
        epoch_val_loss = 0.
        samples_processed = 0
        with torch.no_grad():
            for loaded_data in val_dataloader:

                (data_batch_list, age, stage), label_tensor_list = loaded_data
                current_batch_size = len(data_batch_list[0])
                samples_processed += current_batch_size

                targets = encode_survival(*label_tensor_list, bins=time_intervals)
                targets = torch.reshape(targets, (current_batch_size, -1))
                if use_aux_feats:
                    predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs,
                                                      aux_features=[age, stage])
                else:
                    predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs,
                                                      aux_features=None)
                predictions = torch.reshape(predictions, (current_batch_size, -1))
                # Calculate validation loss for the batch
                loss = mtlr_neg_log_likelihood(predictions, targets, model, C1=config["C1"], average=False)
                epoch_val_loss += loss
        epoch_val_loss /= samples_processed
        val_losses.append(float(epoch_val_loss))
        # print(f"Average validation set loss: {epoch_val_loss:>3f}\n")
        # endregion Evaluate

        # Checkpointing
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            torch.save(
                (t, mp_modules.state_dict(), model.state_dict(), optimizer.state_dict()),
                os.path.join(tmp_ckpt_dir, "checkpoint.pt")
            )
            train.report(
                metrics={
                    "training_loss": float(epoch_train_loss),
                    "validation_loss": float(epoch_val_loss)
                },
                checkpoint=Checkpoint.from_directory(tmp_ckpt_dir)
            )


def load_dataset(graph_dir: str, transform: Optional[Callable] = None) -> CancerGraphDataset:
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    graph_files = sorted(os.listdir(os.path.join(graph_dir, "raw")))  # Raw graph Data object files
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerGraphDataset(root=graph_dir, data_files=graph_files, transform=transform)
    return dataset


def main():
    # region Parse args
    parser = argparse.ArgumentParser(
        description="GNN hyperparameter tuning"
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
        "-i", "--n_intervals",
        help="The number of intervals to split time into for MTLR survival modeling. Intervals are determined from "
             "the n quantiles of survival times of uncensored patients. If n_intervals is set to 2, survival modeling "
             "equates to predicting whether a patient will survive at least until the single time point that separates "
             "the two intervals. If n_intervals is set to 2, 3 years will be used as the cutoff time.",
        type=int
    )
    parser.add_argument(
        "--sagpool",
        help="Use SAGPool layers in the GNN model.",
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

    # # For interactive debugging
    # args = {
    #     "data_dir": "data",
    #     "output_dir": "test_expt",
    #     "cancer_type": "LIHC",
    #     "database": "reactome",
    #     "directed": True,
    #     "relational": False,
    #     "merge_pathways": False,
    #     "sagpool": True,
    #     "batch_size": 48,
    #     "n_intervals": 5,
    #     "use_clin_feats": False,
    #     "normalize_gene_exprs": False,
    # }

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
    use_sagpool = args["sagpool"]
    sagpool = "sagpool" if use_sagpool else "no-sagpool"
    batch_size = args["batch_size"]
    n_intervals = args["n_intervals"]
    tuning_fold = 0  # Fold 0 is reserved exclusively for hyperparameter tuning
    use_clin_feats = True if args["use_clin_feats"] else False
    normalize_gene_exprs = True if args["normalize_gene_exprs"] else False
    # endregion Define important values

    # region Directories
    shared_path_sgmt = os.path.join(database, merge, direction, graph_type)
    cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer_type))
    cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer_type))
    graph_data_dir = os.path.join(cancer_data_dir, "graphs", shared_path_sgmt)
    model_dir = os.path.join(cancer_out_dir, "models", model_type, shared_path_sgmt)
    ckpt_dir = os.path.join(cancer_out_dir, "checkpoints", model_type, shared_path_sgmt)
    export_dir = os.path.join(cancer_out_dir, "exports", model_type, shared_path_sgmt)
    hp_dir = os.path.join(cancer_out_dir, "hyperparameters", model_type, shared_path_sgmt)

    # Check for the existence of directories and create them if necessary
    maybe_create_directories(model_dir, ckpt_dir, export_dir, hp_dir)
    # endregion Directories

    # region Files to read/write
    graph_data_files = sorted(os.listdir(os.path.join(graph_data_dir, "raw")))  # Raw graph Data object files
    # Gene to pathway graph coarsening scheme for pooling graphs if merge_pathways is True
    file_name = f"{cancer_type}_{database}_{direction}_pooling_assignment.pt"  # Only used if merge_pathways
    assignment_matrix_file = os.path.join(data_dir, cancer_type, file_name) if merge_pathways else None
    # HSA/ENTREZ IDs of features (genes) in the GNN input graph
    shared_prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}"
    # Hyperparameters from the current tuning trial
    hp_file = os.path.join(hp_dir, f"{shared_prefix}_{sagpool}_hyperparameters.pkl")
    # endregion Files to read/write

    if assignment_matrix_file is not None:  # Only used if merge_pathways
        with open(assignment_matrix_file, "rb") as file_in:
            buffer = io.BytesIO(file_in.read())
        pooling_assignment = torch.load(buffer)

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(cancer_data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_test_names = pickle.load(file_in)
    # Get the names of the raw Data files for biopsies the current CV fold
    train_names, val_names = train_test_names[tuning_fold]
    # Get the indices of the current CV fold's raw Data files in the file list
    train_idx = [graph_data_files.index(name) for name in train_names]
    val_idx = [graph_data_files.index(name) for name in val_names]
    # Load dataset
    if normalize_gene_exprs:
        ds = load_dataset(graph_data_dir, transform=GraphRangeScaler(attrs=["x"], dim=0))
    else:
        ds = load_dataset(graph_data_dir, transform=None)

    # region Calculate time intervals
    # Iterate over all training examples once to get survival times and censor status for calculating time intervals
    train_survival_times = []
    train_censor_status = []
    # Get training set labels and use them to calculate the time bins for MTLR
    for i in train_idx:
        label_data = ds[i][1]
        train_survival_times.append(label_data[0])
        train_censor_status.append(label_data[1])
    train_survival_times = torch.tensor(train_survival_times)
    train_censor_status = torch.tensor(train_censor_status)
    if n_intervals == 2:  # Single time survival at 3 years
        intervals = torch.tensor([3 * 365.])
    else:
        # Only examples where censor_status == 1 will be used to calculate the time intervals
        intervals = make_time_bins(train_survival_times, num_bins=n_intervals, use_quantiles=True,
                                   event=train_censor_status)
    # endregion Calculate time intervals

    # region Perform hyperparameter tuning
    # If tuning a GNN with SAGPool layers, the GNNs without SAGPool layers must have been tuned already because the
    # SAGPool GNNs are tuned using the regularization/weight decay parameters that were tuned with the non-SAGPooled
    # GNNs.
    if use_sagpool:  # Tune GNN with SAGPool layers
        no_sagpool_hp_file = os.path.join(hp_dir, f"{shared_prefix}_no-sagpool_hyperparameters.pkl")
        if not os.path.exists(no_sagpool_hp_file):
            raise FileNotFoundError(f"Could not find file {no_sagpool_hp_file}. Perform hyperparameter tuning with "
                                    f"non-SAGPool GNN first.")
        with open(no_sagpool_hp_file, "rb") as file_in:
            no_sagpool_hp_dict = pickle.load(file_in)
        hp_dict = {
            "weight_decay": tune.grid_search([no_sagpool_hp_dict["weight_decay"]]),
            "C1": tune.grid_search([no_sagpool_hp_dict["C1"]]),
            "ratio": tune.grid_search([0.6, 0.7, 0.8]),
            "lr": tune.grid_search([no_sagpool_hp_dict["lr"]])
        }
    else:  # Tune GNN without SAGPool layers
        hp_dict = {
            # "weight_decay": tune.grid_search([0.005, 0.0075, 0.01]),  # Too large, doesn't learn
            "weight_decay": tune.grid_search([5e-4, 7.5e-4, 1e-3, 2.5e-3]),
            "C1": tune.grid_search([0.0025, 0.005, 0.0075, 0.01]),
            "ratio": tune.grid_search([0.7]),  # Dummy value; not used unless tuning with SAGPool layers
            "lr": tune.grid_search([5e-4])
        }

    train_model = tune.with_parameters(
        train_loop, dataset=ds, train_indices=train_idx, val_indices=val_idx, batch_size=batch_size,
        use_relational_graphs=relational, use_sagpool=use_sagpool, time_intervals=intervals,
        use_aux_feats=use_clin_feats
    )
    #  Adjust `resources={"cpu": x}` based on available cores to optimize performance. This is the number of
    #  cores that each trial will use. For example, if 12 cores are available and `resources={"cpu": 2}`, six trials
    #  can be run concurrently using two cores each.
    n_cpu = 4
    train_model = tune.with_resources(train_model, resources={"cpu": n_cpu})

    storage_path = os.path.abspath(ckpt_dir)
    expt_name = (f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}_model_feature"
                 f"-selection={sagpool}_tuning")
    expt_storage_path = os.path.join(storage_path, expt_name)
    if tune.Tuner.can_restore(expt_storage_path):
        # Auto-resume experiment after fault occurred or restore completed tuning experiment
        tuner = tune.Tuner.restore(expt_storage_path, trainable=train_model, resume_errored=True)
    else:
        tuner = tune.Tuner(
            train_model,
            param_space=hp_dict,
            run_config=train.RunConfig(
                storage_path=storage_path,
                name=expt_name,
                failure_config=train.FailureConfig(max_failures=1),
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_score_attribute="validation_loss",
                    checkpoint_score_order="min",
                    num_to_keep=5  # Keep the five checkpoints with the lowest validation losses
                )
            ),
        )
    results = tuner.fit()
    best_result = results.get_best_result(metric="validation_loss", mode="min")
    best_hypers = best_result.config  # Best hyperparameters
    with open(hp_file, "wb") as file_out:
        pickle.dump(best_hypers, file_out)
    # endregion Perform hyperparameter tuning

    restored_tuner = tune.Tuner.restore(expt_storage_path, trainable=train_model)  # Restores most recent
    result_grid = restored_tuner.get_results()  # List of tuning trials
    # results_df = result_grid.get_dataframe()
    best_result = result_grid.get_best_result(metric="validation_loss", mode="min")
    best_hypers = best_result.config
    ax = None
    for result in result_grid:
        if use_sagpool:
            label = f"SAGPool node retention rate={result.config['ratio']:.1f}"
        else:
            label = f"weight decay={result.config['weight_decay']:.1e}, C1={result.config['C1']:.1e}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "validation_loss", label=label, figsize=(10, 8))
        else:
            result.metrics_dataframe.plot("training_iteration", "validation_loss", ax=ax, label=label, figsize=(10, 8))
    ax.set_title(f"GNN tuning results: {cancer_type}, learning rate={best_hypers['lr']}, batch size={batch_size}")
    ax.set_ylabel("Validation Loss (Negative Log Likelihood)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    box_pos = ax.get_position()
    ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.7, box_pos.height])
    fig = ax.get_figure()
    fig.savefig(os.path.join(cancer_out_dir, f"{expt_name}_validation-loss.png"))


if __name__ == "__main__":
    main()
