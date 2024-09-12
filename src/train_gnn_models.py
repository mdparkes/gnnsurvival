import argparse
import io
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import tempfile
import torch
import torchmetrics

from filelock import FileLock
from matplotlib.ticker import MaxNLocator
from ray import tune, train, init
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader
from torchmtlr import mtlr_neg_log_likelihood
from torchmtlr.utils import encode_survival, make_time_bins
from typing import Any, Callable, Dict, Optional, Sequence

from dataset_definitions import CancerGraphDataset
from models import NeuralNetworkMTLR, IndividualPathsMPNN
from transformations import GraphRangeScaler
from utilities import maybe_create_directories


def load_dataset(graph_dir: str, transform: Optional[Callable] = None) -> CancerGraphDataset:
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    graph_files = sorted(os.listdir(os.path.join(graph_dir, "raw")))  # Raw graph Data object files
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerGraphDataset(root=graph_dir, data_files=graph_files, transform=transform)
    return dataset


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
        data_dir: str,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        worker_batch_size: int,
        use_relational_graphs: bool,
        use_sagpool: bool,
        time_intervals: Tensor,
        fold: int,
        use_aux_feats: bool,
        use_transform: bool
) -> None:
    # worker_batch_size is the size of the batch subset handled by the worker. The global batch size is calculated as
    # the number of workers times the worker batch size. For example, if the global batch size is 50 and there are 5
    # workers, worker_batch_size should be 10.
    
    epochs = 150

    # region Dataset and DataLoaders
    # Create training and validation fold Dataset objects
    if use_transform:
        dataset = load_dataset(data_dir, transform=GraphRangeScaler(attrs=["x"], dim=0))
    else:
        dataset = load_dataset(data_dir, transform=None)
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    # Create samplers that partition stratified CV folds into disjoint random batches
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=423, drop_last=False)
    val_sampler = DistributedSampler(val_dataset,  shuffle=True, seed=423, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=worker_batch_size, sampler=train_sampler)
    train_dataloader = prepare_data_loader(train_dataloader)
    val_dataloader = DataLoader(val_dataset, batch_size=worker_batch_size, sampler=val_sampler)
    val_dataloader = prepare_data_loader(val_dataloader)
    # endregion Dataset and DataLoaders

    # region Initialize models and optimizer
    n_submodules = len(dataset[0][0][0])  # Number of different pathways
    if use_aux_feats:
        n_aux_feats = len(dataset[0][0]) - 1  # Number of auxiliary features
        total_feats = n_submodules + n_aux_feats
    else:
        total_feats = n_submodules

    mp_modules = torch.nn.ModuleList()
    for i in range(n_submodules):  # Initialize modules for non-relational graphs
        num_nodes = int(dataset[0][0][0][i].x.size(0))  # Number of nodes in the pathway graph
        mp_mod = IndividualPathsMPNN(message_passing="graphsage", use_sagpool=use_sagpool,
                                     ratio=config["ratio"], num_nodes=num_nodes)
        mp_mod = DistributedDataParallel(mp_mod)
        mp_mod = prepare_model(mp_mod)
        mp_modules.append(mp_mod)

    model = NeuralNetworkMTLR(num_time_bins=len(time_intervals), in_features=total_feats)
    model = DistributedDataParallel(model)
    model = prepare_model(model)

    optimizer = torch.optim.Adam([
        {"params": mp_modules.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint_fold={fold}.pt"))
            start_epoch, mp_modules_state, model_state, optimizer_state = ckpt
            mp_modules.load_state_dict(mp_modules_state)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    else:
        start_epoch = 0

    # MeanMetric objects will aggregate the losses across all workers in the DDP process
    mean_train_loss = torchmetrics.MeanMetric()
    mean_valid_loss = torchmetrics.MeanMetric()

    for t in range(start_epoch, epochs):
        # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the
        # DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same
        # ordering will always be used.
        train_sampler.set_epoch(t)
        val_sampler.set_epoch(t)

        # region Train
        mp_modules.train()
        model.train()
        epoch_train_loss = 0.
        n_batches = len(train_dataloader)
        for loaded_data in train_dataloader:

            (data_batch_list, age, stage), label_tensor_list = loaded_data
            current_batch_size = len(data_batch_list[0])
            # data_batch_list is a list of m DataBatch objects, where m is the number of graphs fed through the GNN
            # for a single patient. Each DataBatch object represents a batch of a particular graph.
            # label_tensor_list is a length-2 list. The first item is a shape [n] Tensor of survival times (where n
            # is the number of biopsies in the batch). The second item is a shape [n] Tensor of censor bits.

            targets = encode_survival(*label_tensor_list, bins=time_intervals)
            targets = torch.reshape(targets, (current_batch_size, -1))

            aux_features = [age, stage] if use_aux_feats else None
            predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs, aux_features)
            predictions = torch.reshape(predictions, (current_batch_size, -1))

            wt = current_batch_size / worker_batch_size  # Down-weights the per-patient losses from undersized batches
            loss = wt * mtlr_neg_log_likelihood(predictions, targets, model, C1=config["C1"], average=True)
            epoch_train_loss += loss  # Running total of this epoch's mean per-patient training minibatch losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= n_batches  # Per-patient average training loss for this epoch
        mean_train_loss(epoch_train_loss)  # Save the worker's epoch training loss in the aggregator
        aggregated_train_loss = mean_train_loss.compute().item()  # Aggregate mean loss across workers
        mean_train_loss.reset()  # Reset for next epoch
        # endregion Train

        # region Evaluate
        mp_modules.eval()
        model.eval()
        epoch_val_loss = 0.
        n_batches = len(val_dataloader)
        # samples_processed = 0
        with torch.no_grad():
            for loaded_data in val_dataloader:

                (data_batch_list, age, stage), label_tensor_list = loaded_data
                current_batch_size = len(data_batch_list[0])

                targets = encode_survival(*label_tensor_list, bins=time_intervals)
                targets = torch.reshape(targets, (current_batch_size, -1))

                if use_aux_feats:
                    predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs,
                                                      aux_features=[age, stage])
                else:
                    predictions, _ = gnn_forward_pass(data_batch_list, mp_modules, model, use_relational_graphs,
                                                      aux_features=None)
                predictions = torch.reshape(predictions, (current_batch_size, -1))

                wt = current_batch_size / worker_batch_size
                loss = wt * mtlr_neg_log_likelihood(predictions, targets, model, C1=config["C1"], average=True)
                epoch_val_loss += loss  # Running total of this epoch's mean per-patient validation minibatch losses

        epoch_val_loss /= n_batches  # Per-patient average validation loss for this epoch
        mean_valid_loss(epoch_val_loss)  # Save the worker's epoch validation loss in the aggregator
        aggregated_val_loss = mean_valid_loss.compute().item()  # Aggregate mean loss across workers
        mean_valid_loss.reset()  # Reset for next epoch
        # endregion Evaluate

        # Checkpointing
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            torch.save(
                (t, mp_modules.state_dict(), model.state_dict(), optimizer.state_dict()),
                os.path.join(tmp_ckpt_dir, f"checkpoint_fold={fold}.pt")
            )
            metrics = {
                "training_loss": float(aggregated_train_loss),
                "validation_loss": float(aggregated_val_loss)
            }
            train.report(
                metrics=metrics,
                checkpoint=Checkpoint.from_directory(tmp_ckpt_dir)
            )
        # Report metrics from worker 0
        if train.get_context().get_world_rank() == 0:
            print(f"Epoch {t:>3d} -- Training loss: {metrics['training_loss']:>4f}, Validation loss: "
                  f"{metrics['validation_loss']:>4f}")


def main():
    # region Parse args
    parser = argparse.ArgumentParser(
        description="Train a GNN model of survival with cross-validation"
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
        "-nw", "--num_workers",
        help="The number of workers for distributed training. Should evenly divide batch_size.",
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
    num_workers = args["num_workers"]
    n_intervals = args["n_intervals"]
    use_clin_feats = True if args["use_clin_feats"] else False
    normalize_gene_exprs = True if args["normalize_gene_exprs"] else False
    # endregion Define important values

    if batch_size % num_workers != 0:
        raise ValueError(f"Command line argument \"num_workers\" ({num_workers}) must evenly divide \"batch_size\" ("
                         f"{batch_size})")

    # region Directories
    shared_path_sgmt = os.path.join(database, merge, direction, graph_type)
    cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer_type))
    cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer_type))
    graph_data_dir = os.path.join(cancer_data_dir, "graphs", shared_path_sgmt)
    model_dir = os.path.join(cancer_out_dir, "models", model_type, shared_path_sgmt)
    ckpt_dir = os.path.join(cancer_out_dir, "checkpoints", model_type, shared_path_sgmt)
    hp_dir = os.path.join(cancer_out_dir, "hyperparameters", model_type, shared_path_sgmt)
    export_dir = os.path.join(cancer_out_dir, "exports", model_type, shared_path_sgmt)

    # Check for the existence of directories and create them if necessary
    maybe_create_directories(model_dir, ckpt_dir, export_dir, hp_dir)
    # endregion Directories

    # region Files to read/write
    graph_data_files = sorted(os.listdir(os.path.join(graph_data_dir, "raw")))  # Raw graph Data object files
    # Gene to pathway graph coarsening scheme for pooling graphs if merge_pathways is True
    file_name = f"{cancer_type}_{database}_{direction}_pooling_assignment.pt"  # Only used if merge_pathways
    assignment_matrix_file = os.path.join(data_dir, cancer_type, file_name) if merge_pathways else None

    shared_prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}"
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
    n_folds = len(train_test_names)
    start_fold = 1  # Folds 1 through k - 1 are used for model evaluation

    # Load hyperparameters
    with open(hp_file, "rb") as file_in:
        hp_dict = pickle.load(file_in)

    # Load dataset
    transform = GraphRangeScaler(attrs=["x"], dim=0) if normalize_gene_exprs else None
    ds = load_dataset(graph_data_dir, transform=transform)

    metrics_df_list = []  # Store the results' metrics dataframes in this list
    for k in range(start_fold, n_folds):
        # Get the names of the raw Data files for biopsies the current CV fold
        train_names, val_names = train_test_names[k]
        # Get the indices of the current CV fold's raw Data files in the file list
        train_idx = [graph_data_files.index(name) for name in train_names]
        val_idx = [graph_data_files.index(name) for name in val_names]
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

        storage_path = os.path.abspath(ckpt_dir)
        expt_name = (f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}_model_feature"
                     f"-selection={sagpool}_fold={k}")
        expt_storage_path = os.path.join(storage_path, expt_name)

        init(log_to_driver=False, ignore_reinit_error=True)  # should suppress info messages to stdout but allow logging

        worker_bsize = int(batch_size / num_workers)
        train_model = tune.with_parameters(
            train_loop, data_dir=graph_data_dir, train_indices=train_idx, val_indices=val_idx,
            worker_batch_size=worker_bsize, use_relational_graphs=relational, use_sagpool=use_sagpool,
            time_intervals=intervals, fold=k, use_aux_feats=use_clin_feats, use_transform=normalize_gene_exprs
        )

        if TorchTrainer.can_restore(expt_storage_path):
            # Auto-resume training if fault occurred
            trainer = TorchTrainer.restore(
                expt_storage_path, train_loop_per_worker=train_model, train_loop_config=hp_dict
            )
        else:
            # Begin training
            trainer = TorchTrainer(
                train_model,
                train_loop_config=hp_dict,
                scaling_config=train.ScalingConfig(num_workers=num_workers-1, use_gpu=False),
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
        results = trainer.fit()
        # Save best model
        best_ckpt = results.get_best_checkpoint(metric="validation_loss", mode="min")
        shared_prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}"
        model_file = os.path.join(model_dir, f"{shared_prefix}_model_feature-selection={sagpool}_fold={k}.pt")
        with best_ckpt.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint_fold={k}.pt"))
            _, mp_modules_state, model_state, optimizer_state = ckpt
        torch.save((intervals, mp_modules_state, model_state, optimizer_state), model_file)
        metrics_df_list.append(results.metrics_dataframe)

    # Plot results
    width_ratios = [1. for _ in range(start_fold, n_folds)]
    width_ratios[-1] = 1.5
    fig, axs = plt.subplots(nrows=1, ncols=len(width_ratios), sharex=True, sharey=True, width_ratios=width_ratios,
                            figsize=(3 * n_folds, 4), )
    for k in range(start_fold, n_folds):
        i = k - start_fold  # index of ax in axs, metrics_dataframe in metrics_df_list
        plt_df = metrics_df_list[i]
        sns.lineplot(x="training_iteration", y="validation_loss", label="Test Dataset", data=plt_df,
                     linestyle="solid", ax=axs[i])
        sns.lineplot(x="training_iteration", y="training_loss", label="Training Dataset", data=plt_df,
                     linestyle="dashed", ax=axs[i])
        axs[i].set_title(f"Cross Validation Fold {k}")
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        if k < n_folds - 1:
            axs[i].get_legend().remove()  # Only show the legend on the final subplot
    fig.supxlabel("Training iteration")
    fig.supylabel("Loss (Negative Log Likelihood)", x=0.08)
    axs[-1].legend(loc="center left", bbox_to_anchor=(1, 1))
    box_pos = axs[-1].get_position()
    axs[-1].set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.66, box_pos.height])
    prefix = f"{cancer_type}_{model_type}_{database}_{merge}_{direction}_{graph_type}_model_feature-selection={sagpool}"
    fig.savefig(os.path.join(cancer_out_dir, f"{prefix}_loss.png"))


if __name__ == "__main__":
    main()