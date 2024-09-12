"""
Note:
    Uses sagpool selections from GNNs with non-relational unmerged graphs -- hardcoded behaviour.

"""

import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
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
from typing import Any, Callable, Dict, Optional, Sequence, Union

from dataset_definitions import CancerDataset
from models import NeuralNetworkMTLR, SparseMLP
from transformations import RangeScaler
from utilities import maybe_create_directories


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
        dataset = CancerDataset(root=root, data_files=data_files, transform=transform)  # No transform
    return dataset


def train_loop(
        config: Dict[str, Any],
        *,
        data_dir: str,
        train_names: Sequence[str],
        val_names: Sequence[str],
        worker_batch_size: int,
        pathway_mask: Tensor,
        feature_indices: Optional[Union[Sequence[int], None]],
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
    transform = RangeScaler(dim=0) if use_transform else None
    train_dataset = load_dataset(data_dir, train_names, transform=transform)
    val_dataset = load_dataset(data_dir, val_names, transform=transform)

    # Create samplers that partition stratified CV folds into disjoint random batches
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=423, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=worker_batch_size, sampler=train_sampler)
    train_dataloader = prepare_data_loader(train_dataloader)

    val_sampler = DistributedSampler(val_dataset,  shuffle=True, seed=423, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=worker_batch_size, sampler=val_sampler)
    val_dataloader = prepare_data_loader(val_dataloader)
    # endregion Dataset and DataLoaders

    # Restrict the mask to input genes (columns) that will be used by the sparse MLP
    if feature_indices is not None:
        pathway_mask = pathway_mask[:, feature_indices]

    if use_aux_feats:
        n_aux_feats = len(train_dataset[0][0]) - 1
        total_feats = pathway_mask.shape[0] + n_aux_feats
    else:
        total_feats = pathway_mask.shape[0]

    # region Initialize models and optimizer
    sparse_layer = SparseMLP(pathway_mask)
    sparse_layer = DistributedDataParallel(sparse_layer)
    sparse_layer = prepare_model(sparse_layer)

    model = NeuralNetworkMTLR(num_time_bins=len(time_intervals), in_features=total_feats)
    model = DistributedDataParallel(model)
    model = prepare_model(model)

    optimizer = torch.optim.Adam([
        {"params": sparse_layer.parameters()},
        {"params": model.parameters()}
    ], lr=config["lr"], weight_decay=config["weight_decay"])
    # endregion Initialize models and optimizer

    # Check for and load checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint_fold={fold}.pt"))
            start_epoch, sparse_layer_state, model_state, optimizer_state = ckpt
            sparse_layer.load_state_dict(sparse_layer_state)
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
        sparse_layer.train()
        model.train()
        epoch_train_loss = 0.
        n_batches = len(train_dataloader)
        for loaded_data in train_dataloader:

            (gene_exprs, age, stage), label_tensor_list = loaded_data
            current_batch_size = gene_exprs.shape[0]
            gene_exprs = torch.reshape(gene_exprs, (current_batch_size, -1))
            if feature_indices is not None:
                # Only use the selected genes at the specified indices
                gene_exprs = gene_exprs[:, feature_indices]

            targets = encode_survival(*label_tensor_list, bins=time_intervals)
            targets = torch.reshape(targets, (current_batch_size, -1))

            inputs = sparse_layer(gene_exprs)
            if use_aux_feats:
                age = age.reshape([-1, 1])
                stage = stage.reshape([-1, 1])
                inputs = torch.cat([inputs, age, stage], dim=-1)

            predictions = model(inputs)
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
        sparse_layer.eval()
        model.eval()
        epoch_val_loss = 0.
        n_batches = len(val_dataloader)
        with torch.no_grad():
            for loaded_data in val_dataloader:

                (gene_exprs, age, stage), label_tensor_list = loaded_data
                current_batch_size = gene_exprs.shape[0]
                gene_exprs = torch.reshape(gene_exprs, (current_batch_size, -1))
                if feature_indices is not None:
                    # Only use the selected genes at the specified indices
                    gene_exprs = gene_exprs[:, feature_indices]

                targets = encode_survival(*label_tensor_list, bins=time_intervals)
                targets = torch.reshape(targets, (current_batch_size, -1))

                inputs = sparse_layer(gene_exprs)
                if use_aux_feats:
                    age = age.reshape([-1, 1])
                    stage = stage.reshape([-1, 1])
                    inputs = torch.cat([inputs, age, stage], dim=-1)

                predictions = model(inputs)
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
                (t, sparse_layer.state_dict(), model.state_dict(), optimizer.state_dict()),
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
        description="Train an MLP model of survival with cross-validation"
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
        help="If set, use SAGPool-based feature selection. Requires that SAGPool GNNs are trained first.",
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
    #     "cancer_type": "KIRC",
    #     "database": "reactome",
    #     "directed": True,
    #     "sagpool": False,
    #     "batch_size": 48,
    #     "num_workers": 8,
    #     "n_intervals": 5,
    #     "use_clin_feats": False,
    #     "normalize_gene_exprs": False,
    # }

    # region Define important values
    data_dir = args["data_dir"]  # e.g. ./data
    output_dir = args["output_dir"]  # e.g. ./experiment6
    cancer_type = args["cancer_type"]
    model_type = "mlp"
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
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
    cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer_type))
    cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer_type))
    input_data_dir = os.path.join(cancer_data_dir, "mlp_inputs", database, direction)  # Directory with inputs to MLP

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
    # HSA/ENTREZ IDs of features (genes) in the MLP input tensor
    file_name = f"{cancer_type}_{database}_{direction}_mlp_feature_names.pkl"
    feature_names_file = os.path.join(cancer_data_dir, file_name)
    # HSA IDs of pathways in the first hidden layer of the MLP
    file_name = f"{cancer_type}_{database}_{direction}_mlp_pathway_names.npy"
    pathway_names_file = os.path.join(cancer_data_dir, file_name)
    hp_file = os.path.join(hp_dir, f"{cancer_type}_{model_type}_{database}_{direction}_hyperparameters.pkl")

    # Load feature names in the order they appear in the unfiltered input to the SparseMLP
    with open(feature_names_file, "rb") as file_in:
        all_feature_names = pickle.load(file_in)

    # Load weight mask for SparseMLP Module
    with open(mask_matrix_file, "rb") as file_in:
        buffer = io.BytesIO(file_in.read())
    mlp_mask_matrix = torch.load(buffer)

    # Load pathway names
    pathway_names = np.load(pathway_names_file, allow_pickle=True)
    assert (len(pathway_names) == mlp_mask_matrix.shape[0])

    # Load tuned MLP hyperparameters
    with open(hp_file, "rb") as file_in:
        hp_dict = pickle.load(file_in)

    # Load lists that name the biopsies in each cross validation partition
    with open(os.path.join(cancer_data_dir, "train_test_split_names.pkl"), "rb") as file_in:
        # A list of one tuple per CV fold: (train_names, test_names)
        train_test_names = pickle.load(file_in)
    n_folds = len(train_test_names)
    start_fold = 1  # Folds 1 through k - 1 are used for model evaluation

    # Load dataset
    transform = RangeScaler(dim=0) if normalize_gene_exprs else None
    ds = load_dataset(input_data_dir, transform=transform)

    metrics_df_list = []
    for k in range(start_fold, n_folds):
        # Get the names of the raw Data files for biopsies the current CV fold
        train_names, val_names = train_test_names[k]
        # Get the indices of the current CV fold's raw Data files in the file list
        train_idx = [input_data_files.index(name) for name in train_names]

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

        # Features to use
        if use_sagpool:
            # Gene retention frequencies normalized by the number of times each gene appears in a pathway
            shared_path_sgmt = os.path.join(database, "unmerged", direction, "nonrelational")
            file_name = f"normalized_sagpool_gene_retention_frequencies_{k}.pkl"
            normalized_freq_file = os.path.join(cancer_out_dir, "models", "gnn", shared_path_sgmt, file_name)
            with open(normalized_freq_file, "rb") as file_in:
                normalized_freq = pickle.load(file_in)
            # Get the SAGPool retention rate that was used in the GNNs that generated the frequencies
            gnn_hp_dir = os.path.join(cancer_out_dir, "hyperparameters", "gnn", shared_path_sgmt)
            shared_prefix = f"{cancer_type}_gnn_{database}_unmerged_{direction}_nonrelational"
            gnn_hp_file = os.path.join(gnn_hp_dir, f"{shared_prefix}_{sagpool}_hyperparameters.pkl")
            with open(gnn_hp_file, "rb") as file_in:
                gnn_hp_dict = pickle.load(file_in)
            sagpool_ratio = gnn_hp_dict["ratio"]
            # Calculate the number of features to use from the SAGPool retention rate
            n_features_used = (sagpool_ratio ** 3) * len(all_feature_names[0])  # because there were 3 SAGPool layers
            n_features_used = int(np.ceil(n_features_used))
            # Genes are already in descending order of selection frequency
            features_to_use = list(normalized_freq.keys())[0:n_features_used]
            features_used_idx = [all_feature_names[0].index(feat) for feat in features_to_use]
        else:
            features_used_idx = None  # When this is passed to train_loop `features` param, all features will be used

        storage_path = os.path.abspath(ckpt_dir)
        expt_name = f"{cancer_type}_{model_type}_{database}_{direction}_model_feature-selection={sagpool}_fold={k}"
        expt_storage_path = os.path.join(storage_path, expt_name)

        init(log_to_driver=False, ignore_reinit_error=True)  # should suppress info messages to stdout but allow logging

        worker_bsize = int(batch_size / num_workers)
        train_model = tune.with_parameters(
            train_loop, data_dir=input_data_dir, train_names=train_names, val_names=val_names,
            worker_batch_size=worker_bsize, pathway_mask=mlp_mask_matrix, feature_indices=features_used_idx,
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
        shared_prefix = f"{cancer_type}_{model_type}_{database}_{direction}"
        model_file = os.path.join(model_dir, f"{shared_prefix}_model_feature-selection={sagpool}_fold={k}.pt")
        with best_ckpt.as_directory() as checkpoint_dir:
            ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint_fold={k}.pt"))
            _, sparse_layer_state, model_state, optimizer_state = ckpt
        torch.save((intervals, features_used_idx, sparse_layer_state, model_state, optimizer_state), model_file)
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
    prefix = f"{cancer_type}_{model_type}_{database}_{direction}_model_feature-selection={sagpool}"
    fig.savefig(os.path.join(cancer_out_dir, f"{prefix}_loss.png"))


if __name__ == "__main__":
    main()
