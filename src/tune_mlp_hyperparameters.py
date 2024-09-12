"""
Perform hyperparameter tuning using a fold of data that is exclusively reserved for this purpose.
MLP models can differ in the type of database the pathways were derived from (KEGG/BRITE, Reactome) and whether the
counterpart GNN was over graphs that had only directed edges or both directed and undirected edges. If a GNN only
used graphs with directed edges, its node set only contains genes that participate in directed edges. If a GNN used
both directed and undirected edges, its node set contains genes that participate in either directed or undirected
edges. In both styles of GNN, genes that do not participate in any edge are absent from the node set. Since each MLP
counterpart to a GNN must have exactly the same genes as input, the MLP input depends on the directedness of the
graphs passed through the GNN.
"""

import argparse
import io
import numpy as np
import os
import pickle
import tempfile
import torch

from filelock import FileLock
from matplotlib.ticker import MaxNLocator
from ray import tune, train
from ray.train import Checkpoint
from torch import Tensor
from torch.utils.data import BatchSampler, Dataset, SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torchmtlr import mtlr_neg_log_likelihood
from torchmtlr.utils import encode_survival, make_time_bins
from typing import Any, Callable, Dict, Optional, Sequence

from dataset_definitions import CancerDataset
from models import NeuralNetworkMTLR, SparseMLP
from transformations import RangeScaler
from utilities import maybe_create_directories


def load_dataset(root: str, transform: Optional[Callable] = None) -> CancerDataset:
    """
    Loads a dataset from serialized data on disk.

    :param root: The path to the directory where the data are stored. Should contain a "raw" subdirectory containing
    the serialized data files.
    :param transform: A transformation to apply to the loaded data
    :return: A CancerDataset
    """
    # If merging all pathways into a single large graph, the standardization occurs over all genes in all pathways.
    # If feeding one pathway graph through the NN at a time, standardization is isolated to the pathway's genes.
    data_files = sorted(os.listdir(os.path.join(root, "raw")))  # Raw graph Data object files
    # transformation = StandardizeFeatures(correction=1)  # No transform
    # Use FileLock to make DataLoader threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CancerDataset(root=root, data_files=data_files, transform=transform)  # No transform
    return dataset


def train_loop(
        config: Dict[str, Any],
        *,
        dataset: Dataset,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        batch_size: int,
        pathway_mask: Tensor,
        time_intervals: Tensor,
        use_aux_feats: bool
) -> None:

    epochs = 150

    # region Create dataloaders
    # Create samplers that partition stratified CV folds into disjoint random batches
    train_batch_sampler = BatchSampler(SubsetRandomSampler(train_indices), batch_size, drop_last=False)
    val_batch_sampler = BatchSampler(SubsetRandomSampler(val_indices), batch_size, drop_last=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_sampler=val_batch_sampler, num_workers=4)
    # endregion Create dataloaders

    # region Initialize models and optimizer
    if use_aux_feats:
        n_aux_feats = len(dataset[0][0]) - 1
        total_feats = pathway_mask.shape[0] + n_aux_feats
    else:
        total_feats = pathway_mask.shape[0]

    sparse_layer = SparseMLP(pathway_mask)

    model = NeuralNetworkMTLR(num_time_bins=len(time_intervals), in_features=total_feats)

    optimizer = torch.optim.Adam([
        {"params": sparse_layer.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch, sparse_layer_state, model_state, optimizer_state = ckpt
            sparse_layer.load_state_dict(sparse_layer_state)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    else:
        start_epoch = 0

    for t in range(start_epoch, epochs):

        # print(f"Epoch {t + 1}\n{'':-<80}")

        # region Train
        sparse_layer.train()
        model.train()
        epoch_train_loss = 0.
        samples_processed = 0
        for loaded_data in train_dataloader:

            (gene_exprs, age, stage), label_tensor_list = loaded_data
            current_batch_size = gene_exprs.shape[0]
            samples_processed += current_batch_size

            # label_tensor_list is a length-2 list. The first item is a shape [n] Tensor of survival times (where n
            # is the number of biopsies in the batch). The second item is a shape [n] Tensor of censor bits.
            targets = encode_survival(*label_tensor_list, bins=time_intervals)
            targets = torch.reshape(targets, (current_batch_size, -1))
            inputs = sparse_layer(gene_exprs)
            if use_aux_feats:
                age = age.reshape([-1, 1])
                stage = stage.reshape([-1, 1])
                inputs = torch.cat([inputs, age, stage], dim=-1)
            predictions = model(inputs)
            predictions = torch.reshape(predictions, (current_batch_size, -1))
            # Calculate training loss for the batch
            loss = mtlr_neg_log_likelihood(predictions, targets, model, C1=config["C1"], average=False)
            epoch_train_loss += loss  # Running total of this epoch's training loss
            loss /= current_batch_size  # Per-patient loss for current batch
            # Optimize model weights w.r.t. per-patient batch loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= samples_processed  # Per-patient average training loss for this epoch
        # endregion Train

        # region Evaluate
        sparse_layer.eval()
        model.eval()
        epoch_val_loss = 0.
        samples_processed = 0
        with torch.no_grad():
            for loaded_data in val_dataloader:

                (gene_exprs, age, stage), label_tensor_list = loaded_data
                current_batch_size = gene_exprs.shape[0]
                samples_processed += current_batch_size

                targets = encode_survival(*label_tensor_list, bins=time_intervals)
                targets = torch.reshape(targets, (current_batch_size, -1))
                inputs = sparse_layer(gene_exprs)
                if use_aux_feats:
                    age = age.reshape([-1, 1])
                    stage = stage.reshape([-1, 1])
                    inputs = torch.cat([inputs, age, stage], dim=-1)
                predictions = model(inputs)
                predictions = torch.reshape(predictions, (current_batch_size, -1))
                # Calculate validation loss for the batch
                loss = mtlr_neg_log_likelihood(predictions, targets, model, C1=config["C1"], average=False)
                epoch_val_loss += loss  # Running total of this epoch's validation loss

        epoch_val_loss /= samples_processed
        # endregion Evaluate

        # Checkpointing
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            torch.save(
                (t, sparse_layer.state_dict(), model.state_dict(), optimizer.state_dict()),
                os.path.join(tmp_ckpt_dir, "checkpoint.pt")
            )
            train.report(
                metrics={
                    "training_loss": float(epoch_train_loss),
                    "validation_loss": float(epoch_val_loss)
                },
                checkpoint=Checkpoint.from_directory(tmp_ckpt_dir)
            )


def main():

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Tunes the hyperparameters of a sparse MLP model of cancer survival"
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
        help="If the --sagpool flag is set, the --directed flag ensures that the sagpool-based gene selections are "
             "taken from a GNN model trained on graphs that only use directed edges. The default behavior is to take "
             "gene selections from a model trained on graphs that have both directed and undirected edges.",
        action="store_true"
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
    model_type = "mlp"
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
    # use_sagpool = False
    # sagpool = "sagpool" if use_sagpool else "no-sagpool"
    batch_size = args["batch_size"]
    n_intervals = args["n_intervals"]
    tuning_fold = 0  # Fold 0 is reserved exclusively for hyperparameter tuning
    use_clin_feats = True if args["use_clin_feats"] else False
    normalize_gene_exprs = True if args["normalize_gene_exprs"] else False
    # endregion Define important values

    # region Directories
    cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer_type))
    cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer_type))
    input_data_dir = os.path.join(cancer_data_dir, "mlp_inputs", database, direction)  # Directory with inputs to MLP
    # If SAGPool-based pooling is used, a different MLP will be tuned per GNN that used SAGPool because the SAGPool
    # selections may differ between GNN models
    shared_path_sgmt = os.path.join(model_type, database, direction)
    model_dir = os.path.join(cancer_out_dir, "models", shared_path_sgmt)
    ckpt_dir = os.path.join(cancer_out_dir, "checkpoints", shared_path_sgmt)
    hp_dir = os.path.join(cancer_out_dir, "hyperparameters", shared_path_sgmt)
    export_dir = os.path.join(cancer_out_dir, "exports", shared_path_sgmt)
    # Check for the existence of directories and create them if necessary
    maybe_create_directories(model_dir, ckpt_dir, export_dir, hp_dir)
    # endregion Directories

    # region Files to read/write
    input_data_files = sorted(os.listdir(os.path.join(input_data_dir, "raw")))  # Raw graph Data object files
    # Weight mask for first hidden layer of MLP
    file_name = f"{cancer_type}_{database}_{direction}_mlp_mask.pt"
    mask_matrix_file = os.path.join(data_dir, cancer_type, file_name)
    # HSA IDs of pathways in the first hidden layer of the MLP
    file_name = f"{cancer_type}_{database}_{direction}_mlp_pathway_names.npy"
    pathway_names_file = os.path.join(cancer_data_dir, file_name)
    hp_file = os.path.join(hp_dir, f"{cancer_type}_{model_type}_{database}_{direction}_hyperparameters.pkl")
    # endregion Files to read/write

    # Load weight mask for SparseMLP Module
    with open(mask_matrix_file, "rb") as file_in:
        buffer = io.BytesIO(file_in.read())
    mlp_mask_matrix = torch.load(buffer)
    # Load pathway names
    pathway_names = np.load(pathway_names_file, allow_pickle=True)
    assert(len(pathway_names) == mlp_mask_matrix.shape[0])

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(cancer_data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_test_names = pickle.load(file_in)

    # Get the names of the raw Data files for biopsies the current CV fold
    train_names, val_names = train_test_names[tuning_fold]
    # Get the indices of the current CV fold's raw Data files in the file list
    train_idx = [input_data_files.index(name) for name in train_names]
    val_idx = [input_data_files.index(name) for name in val_names]
    # Load dataset
    if normalize_gene_exprs:
        ds = load_dataset(input_data_dir, transform=RangeScaler(dim=0))
    else:
        ds = load_dataset(input_data_dir, transform=None)

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
    # Hyperparameter options
    hp_dict = {
        "weight_decay": tune.grid_search([0.0025, 0.005, 0.0075, 0.01]),
        "C1": tune.grid_search([0.0025, 0.005, 0.0075, 0.01]),
        "lr": tune.grid_search([5e-4])
    }
    train_model = tune.with_parameters(
        train_loop, dataset=ds, train_indices=train_idx, val_indices=val_idx, batch_size=batch_size,
        pathway_mask=mlp_mask_matrix, time_intervals=intervals, use_aux_feats=use_clin_feats
    )

    storage_path = os.path.abspath(ckpt_dir)
    expt_name = f"{cancer_type}_{model_type}_{database}_{direction}_model_feature-selection=no-sagpool_tuning"
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

    restored_tuner = tune.Tuner.restore(expt_storage_path, trainable=train_model)
    result_grid = restored_tuner.get_results()
    results_df = result_grid.get_dataframe()

    ax = None
    for result in result_grid:
        label = f"weight decay={result.config['weight_decay']:.1e}, C1={result.config['C1']:.1e}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "validation_loss", label=label, figsize=(10, 8))
        else:
            result.metrics_dataframe.plot("training_iteration", "validation_loss", ax=ax, label=label, figsize=(10, 8))
    ax.set_title(f"MLP tuning results: {cancer_type}, learning rate={best_hypers['lr']}, batch size={batch_size}")
    ax.set_ylabel("Validation Loss (Negative Log Likelihood)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    box_pos = ax.get_position()
    ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.7, box_pos.height])
    fig = ax.get_figure()
    fig.savefig(os.path.join(cancer_out_dir, f"{expt_name}_validation-loss.png"))


if __name__ == "__main__":
    main()