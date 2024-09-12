import argparse
import numpy as np
import os
import pandas as pd

from itertools import repeat
from ray import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a GNN model of survival with cross-validation"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path of the directory where experimental results will be written",
        type=str
    )
    args = vars(parser.parse_args())

    # region Define important values
    data_dir = "data"
    output_dir = args["output_dir"]  # e.g. ./experiment5
    database = "reactome"
    graph_type = "nonrelational"
    relational = False
    direction = "directed"
    directed = True
    merge_pathways = False
    merge = "unmerged"
    batch_size = 48
    num_workers = 4
    n_folds = 6
    start_fold = 1  # Folds 1 through k - 1 are used for model evaluation
    expt_num = int(output_dir[-1])
    if expt_num == 1:
        cancer_types = ["BLCA", "COAD", "GBM", "HNSC", "KIRC", "LGG", "LIHC", "LUAD", "LUSC", "OV", "SKCM", "STAD"]
        n_intervals = 5
        use_clinical_feats = False
        normalize_gene_exprs = False
    elif expt_num == 2:
        cancer_types = ["KIRC", "LGG", "LUAD", "SKCM"]
        n_intervals = 5
        use_clinical_feats = True
        normalize_gene_exprs = False
    elif expt_num == 3:
        cancer_types = ["KIRC", "LGG", "LUAD", "SKCM"]
        n_intervals = 5
        use_clinical_feats = True
        normalize_gene_exprs = True
    elif expt_num == 4:
        cancer_types = ["KIRC", "LGG", "LUAD", "SKCM"]
        n_intervals = 2
        use_clinical_feats = True
        normalize_gene_exprs = True
    elif expt_num == 5:
        cancer_types = ["KIRC", "LGG", "LUAD", "SKCM"]
        n_intervals = 2
        use_clinical_feats = True
        normalize_gene_exprs = True
    else:
        raise ValueError
    # endregion Define important values
    
    for cancer in cancer_types:
        
        losses = {
            "gnn_train_sagpool_fold_1": [],
            "gnn_train_sagpool_fold_2": [],
            "gnn_train_sagpool_fold_3": [],
            "gnn_train_sagpool_fold_4": [],
            "gnn_train_sagpool_fold_5": [],
            "gnn_test_sagpool_fold_1": [],
            "gnn_test_sagpool_fold_2": [],
            "gnn_test_sagpool_fold_3": [],
            "gnn_test_sagpool_fold_4": [],
            "gnn_test_sagpool_fold_5": [],
            "mlp_train_sagpool_fold_1": [],
            "mlp_train_sagpool_fold_2": [],
            "mlp_train_sagpool_fold_3": [],
            "mlp_train_sagpool_fold_4": [],
            "mlp_train_sagpool_fold_5": [],
            "mlp_test_sagpool_fold_1": [],
            "mlp_test_sagpool_fold_2": [],
            "mlp_test_sagpool_fold_3": [],
            "mlp_test_sagpool_fold_4": [],
            "mlp_test_sagpool_fold_5": [],
            "gnn_train_no_sagpool_fold_1": [],
            "gnn_train_no_sagpool_fold_2": [],
            "gnn_train_no_sagpool_fold_3": [],
            "gnn_train_no_sagpool_fold_4": [],
            "gnn_train_no_sagpool_fold_5": [],
            "gnn_test_no_sagpool_fold_1": [],
            "gnn_test_no_sagpool_fold_2": [],
            "gnn_test_no_sagpool_fold_3": [],
            "gnn_test_no_sagpool_fold_4": [],
            "gnn_test_no_sagpool_fold_5": [],
            "mlp_train_no_sagpool_fold_1": [],
            "mlp_train_no_sagpool_fold_2": [],
            "mlp_train_no_sagpool_fold_3": [],
            "mlp_train_no_sagpool_fold_4": [],
            "mlp_train_no_sagpool_fold_5": [],
            "mlp_test_no_sagpool_fold_1": [],
            "mlp_test_no_sagpool_fold_2": [],
            "mlp_test_no_sagpool_fold_3": [],
            "mlp_test_no_sagpool_fold_4": [],
            "mlp_test_no_sagpool_fold_5": [],
        }

        losses_2 = {
            "model_type": [],  # "gnn" or "mlp"
            "train_or_test": [],  # "train" or "test"
            "feature_selection": [],  # "feature selection" or "no feature selection",
            "fold": [],  # integer corresponding to the CV fold
            "epoch": [],  # integer
            "loss": [],  # loss value
        }

        
        # region GNN models
        model_type = "gnn"

        # region Directories
        shared_segment = os.path.join(database, merge, direction, graph_type)
        cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer))
        cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer))
        ckpt_dir = os.path.join(cancer_out_dir, "checkpoints", model_type, shared_segment)
        export_dir = os.path.join(cancer_out_dir, "exports", model_type, shared_segment)
        # endregion Directories

        for k in range(start_fold, n_folds):

            # region Models with feature selection
            use_feat_sel = True
            sagpool = "sagpool"
            expt_name = (f"{cancer}_{model_type}_{database}_{merge}_{direction}_{graph_type}_model_feature"
                         f"-selection={sagpool}_fold={k}")
            storage_path = os.path.abspath(os.path.join(ckpt_dir, expt_name))
            ckpt_paths = []
            for file_name in os.listdir(storage_path):
                file_path = os.path.join(storage_path, file_name)
                if os.path.isdir(file_path):
                    ckpt_paths.append(file_path)
            mtimes = [os.path.getmtime(file_path) for file_path in ckpt_paths]
            expt_storage_path = ckpt_paths[np.argmax(mtimes)]  # Most recently modified checkpoint directory
            results = train.Result.from_path(expt_storage_path)
            losses[f"{model_type}_train_sagpool_fold_{k}"] = results.metrics_dataframe.training_loss.to_list()
            losses[f"{model_type}_test_sagpool_fold_{k}"] = results.metrics_dataframe.validation_loss.to_list()

            n_losses = len(results.metrics_dataframe.training_loss)
            losses_2["loss"].extend(results.metrics_dataframe.training_loss.to_list())
            losses_2["train_or_test"].extend(list(repeat("Train", n_losses)))
            losses_2["feature_selection"].extend(list(repeat("Feature selection", n_losses)))
            losses_2["fold"].extend(list(repeat(k, n_losses)))
            losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
            losses_2["epoch"].extend(list(range(150)))

            n_losses = len(results.metrics_dataframe.validation_loss)
            losses_2["loss"].extend(results.metrics_dataframe.validation_loss.to_list())
            losses_2["train_or_test"].extend(list(repeat("Test", n_losses)))
            losses_2["feature_selection"].extend(list(repeat("Feature selection", n_losses)))
            losses_2["fold"].extend(list(repeat(k, n_losses)))
            losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
            losses_2["epoch"].extend(list(range(150)))
            # endregion Models with feature selection

            # region Models without feature selection
            if expt_num == 1:
                use_feat_sel = False
                sagpool = "no-sagpool"
                expt_name = (f"{cancer}_{model_type}_{database}_{merge}_{direction}_{graph_type}_model_feature"
                             f"-selection={sagpool}_fold={k}")
                storage_path = os.path.abspath(os.path.join(ckpt_dir, expt_name))
                ckpt_paths = []
                for file_name in os.listdir(storage_path):
                    file_path = os.path.join(storage_path, file_name)
                    if os.path.isdir(file_path):
                        ckpt_paths.append(file_path)
                mtimes = [os.path.getmtime(file_path) for file_path in ckpt_paths]
                expt_storage_path = ckpt_paths[np.argmax(mtimes)]  # Most recently modified checkpoint directory
                results = train.Result.from_path(expt_storage_path)
                losses[f"{model_type}_train_no_sagpool_fold_{k}"] = results.metrics_dataframe.training_loss.to_list()
                losses[f"{model_type}_test_no_sagpool_fold_{k}"] = results.metrics_dataframe.validation_loss.to_list()

                n_losses = len(results.metrics_dataframe.training_loss)
                losses_2["loss"].extend(results.metrics_dataframe.training_loss.to_list())
                losses_2["train_or_test"].extend(list(repeat("Train", n_losses)))
                losses_2["feature_selection"].extend(list(repeat("No feature selection", n_losses)))
                losses_2["fold"].extend(list(repeat(k, n_losses)))
                losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
                losses_2["epoch"].extend(list(range(150)))

                n_losses = len(results.metrics_dataframe.validation_loss)
                losses_2["loss"].extend(results.metrics_dataframe.validation_loss.to_list())
                losses_2["train_or_test"].extend(list(repeat("Test", n_losses)))
                losses_2["feature_selection"].extend(list(repeat("No feature selection", n_losses)))
                losses_2["fold"].extend(list(repeat(k, n_losses)))
                losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
                losses_2["epoch"].extend(list(range(150)))
            # endregion Models without feature selection

        # endregion GNN models
    
    
        # region MLP models
        model_type = "mlp"

        # region Directories
        shared_segment = os.path.join(model_type, database, direction)
        cancer_data_dir = os.path.abspath(os.path.join(data_dir, cancer))
        cancer_out_dir = os.path.abspath(os.path.join(output_dir, cancer))
        ckpt_dir = os.path.join(cancer_out_dir, "checkpoints", shared_segment)
        export_dir = os.path.join(cancer_out_dir, "exports", shared_segment)
        # endregion Directories

        # Load most recent checkpointed model
        for k in range(start_fold, n_folds):

            # region Models with feature selection
            sagpool = "sagpool"
            expt_name = f"{cancer}_{model_type}_{database}_{direction}_model_feature-selection={sagpool}_fold={k}"
            storage_path = os.path.abspath(os.path.join(ckpt_dir, expt_name))
            ckpt_paths = []
            for file_name in os.listdir(storage_path):
                file_path = os.path.join(storage_path, file_name)
                if os.path.isdir(file_path):
                    ckpt_paths.append(file_path)
            mtimes = [os.path.getmtime(file_path) for file_path in ckpt_paths]
            expt_storage_path = ckpt_paths[np.argmax(mtimes)]  # Most recently modified checkpoint directory
            results = train.Result.from_path(expt_storage_path)
            losses[f"{model_type}_train_sagpool_fold_{k}"] = results.metrics_dataframe.training_loss.to_list()
            losses[f"{model_type}_test_sagpool_fold_{k}"] = results.metrics_dataframe.validation_loss.to_list()

            n_losses = len(results.metrics_dataframe.training_loss)
            losses_2["loss"].extend(results.metrics_dataframe.training_loss.to_list())
            losses_2["train_or_test"].extend(list(repeat("Train", n_losses)))
            losses_2["feature_selection"].extend(list(repeat("Feature selection", n_losses)))
            losses_2["fold"].extend(list(repeat(k, n_losses)))
            losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
            losses_2["epoch"].extend(list(range(150)))

            n_losses = len(results.metrics_dataframe.validation_loss)
            losses_2["loss"].extend(results.metrics_dataframe.validation_loss.to_list())
            losses_2["train_or_test"].extend(list(repeat("Test", n_losses)))
            losses_2["feature_selection"].extend(list(repeat("Feature selection", n_losses)))
            losses_2["fold"].extend(list(repeat(k, n_losses)))
            losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
            losses_2["epoch"].extend(list(range(150)))
            # endregion Models with feature selection

            # region Models without feature selection
            sagpool = "no-sagpool"
            expt_name = f"{cancer}_{model_type}_{database}_{direction}_model_feature-selection={sagpool}_fold={k}"
            storage_path = os.path.abspath(os.path.join(ckpt_dir, expt_name))
            ckpt_paths = []
            for file_name in os.listdir(storage_path):
                file_path = os.path.join(storage_path, file_name)
                if os.path.isdir(file_path):
                    ckpt_paths.append(file_path)
            mtimes = [os.path.getmtime(file_path) for file_path in ckpt_paths]
            expt_storage_path = ckpt_paths[np.argmax(mtimes)]  # Most recently modified checkpoint directory
            results = train.Result.from_path(expt_storage_path)
            losses[f"{model_type}_train_no_sagpool_fold_{k}"] = results.metrics_dataframe.training_loss.to_list()
            losses[f"{model_type}_test_no_sagpool_fold_{k}"] = results.metrics_dataframe.validation_loss.to_list()

            n_losses = len(results.metrics_dataframe.training_loss)
            losses_2["loss"].extend(results.metrics_dataframe.training_loss.to_list())
            losses_2["train_or_test"].extend(list(repeat("Train", n_losses)))
            losses_2["feature_selection"].extend(list(repeat("No feature selection", n_losses)))
            losses_2["fold"].extend(list(repeat(k, n_losses)))
            losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
            losses_2["epoch"].extend(list(range(150)))

            n_losses = len(results.metrics_dataframe.validation_loss)
            losses_2["loss"].extend(results.metrics_dataframe.validation_loss.to_list())
            losses_2["train_or_test"].extend(list(repeat("Test", n_losses)))
            losses_2["feature_selection"].extend(list(repeat("No feature selection", n_losses)))
            losses_2["fold"].extend(list(repeat(k, n_losses)))
            losses_2["model_type"].extend(list(repeat(model_type.upper(), n_losses)))
            losses_2["epoch"].extend(list(range(150)))
            # endregion Models without feature selection

        # endregion MLP models

        # Export losses dataframe
        losses_df = pd.DataFrame.from_dict(losses, orient="index").T
        file_path = os.path.abspath(os.path.join(output_dir, cancer, "exports", f"{cancer}_losses.csv"))
        losses_df.to_csv(file_path)

        losses_2_df = pd.DataFrame.from_dict(losses_2, orient="index").T
        file_path = os.path.abspath(os.path.join(output_dir, cancer, "exports", f"{cancer}_losses_2.csv"))
        losses_2_df.to_csv(file_path)
