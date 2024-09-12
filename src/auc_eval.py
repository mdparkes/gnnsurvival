"""
Calculates AUC of 3-year survival models using the methods of Liang et al. Patients who were death-censored before 3
years are excluded from the calculation, and all remaining patients are considered uncensored for the purpose of
calculating the AUC score. The predicted score for the AUC calculation is the probability that the patient survived
for at least 1095 days (3 years), and the labels are binary.
"""

import argparse
import os
import numpy as np
import pandas as pd
import re

from sklearn.metrics import roc_auc_score


if __name__ == "__main__":

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Calculate AUC scores of 3-year survival models"
    )
    parser.add_argument(
        "-d", "--data_dir",
        help="The path of the directory where data necessary for training the models are located",
        type=str
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path of the directory where experiment results are located",
        type=str
    )
    parser.add_argument(
        "-t", "--cancer_types",
        help="TCGA letter codes of cancer types whose models will be evaluated",
        nargs="+",
        default=[]
    )
    args = vars(parser.parse_args())
    # endregion Parse args

    data_dir = args["data_dir"]
    output_dir = args["output_dir"]
    cancer_types = args["cancer_types"]
    database = "reactome"
    merge_pathways = False
    merge = "unmerged"
    directed = True
    direction = "directed"
    graph_type = "nonrelational"
    relational = False
    n_intervals = 2
    # endregion Directories

    n_folds = 5  # Does not include the tuning fold
    model_types = ["MLP - No feature selection",
                   "MLP - SAGPool-based feature selection",
                   "GNN - SAGPool"]

    # region Load data from disk
    mlp_no_fs_files = dict()
    mlp_no_fs_data = dict()
    mlp_no_fs_evaluations = dict()

    mlp_fs_files = dict()
    mlp_fs_data = dict()
    mlp_fs_evaluations = dict()

    gnn_fs_files = dict()
    gnn_fs_data = dict()
    gnn_fs_evaluations = dict()

    # For each cancer type keyed in the dictionaries defined above, define a new dict with the following keys
    field_names = ["mtlr_times", "test_curves", "test_surv_times", "test_censor_bits", "train_surv_times",
                   "train_censor_bits"]
    # Regular expression matches results from models without SAGPool-based feature selection
    no_fs_rx = ["no-sagpool_time_points", "no-sagpool_test_predictions", "no-sagpool_time_test",
                "no-sagpool_cns_test", "no-sagpool_time_train", "no-sagpool_cns_train"]
    # Regular expression matches results from models with SAGPool-based feature selection
    fs_rx = ["_sagpool_time_points", "_sagpool_test_predictions", "_sagpool_time_test",
             "_sagpool_cns_test", "_sagpool_time_train", "_sagpool_cns_train"]

    for cancer in cancer_types:

        export_dir = os.path.join(output_dir, cancer, "exports")
        hp_dir = os.path.join(output_dir, cancer, "hyperparameters")
        weights_dir = os.path.join(output_dir, cancer, "weights")

        mlp_shared_segment = f"mlp/{database}/{direction}"
        mlp_ex_dir = os.path.join(output_dir, cancer, "exports", mlp_shared_segment)
        mlp_hp_dir = os.path.join(output_dir, cancer, "hyperparameters", mlp_shared_segment)
        mlp_wt_dir = os.path.join(output_dir, cancer, "weights", mlp_shared_segment)

        gnn_shared_segment = f"gnn/{database}/{merge}/{direction}/{graph_type}"
        gnn_ex_dir = os.path.join(output_dir, cancer, "exports", gnn_shared_segment)
        gnn_hp_dir = os.path.join(output_dir, cancer, "hyperparameters", gnn_shared_segment)
        gnn_wt_dir = os.path.join(output_dir, cancer, "weights", gnn_shared_segment)

        # region MLP without SAGPool-based feature selection
        files = [sorted([os.path.join(mlp_ex_dir, f) for f in os.listdir(mlp_ex_dir) if re.search(rx, f)]) for
                 rx in no_fs_rx]
        mlp_no_fs_files[cancer] = dict(zip(field_names, files))
        mlp_no_fs_data[cancer] = dict(zip(
            field_names,
            [[np.genfromtxt(f, delimiter=",") for f in files] for files in mlp_no_fs_files[cancer].values()]
        ))
        aucs = []
        for k in range(n_folds):
            # Exclude patients that were censored before 3 years
            test_surv_times = mlp_no_fs_data[cancer]["test_surv_times"][k]
            test_censor_bits = mlp_no_fs_data[cancer]["test_censor_bits"][k]
            test_curves = mlp_no_fs_data[cancer]["test_curves"][k]
            sel = test_surv_times >= 3 * 365
            test_censor_bits[sel] = 1.
            sel = test_censor_bits == 1.
            test_surv_times = test_surv_times[sel]
            test_scores = test_curves[sel, 1]  # Probability of surviving >= 3* 365 days
            # Need true labels where 1 => p/t died after 3 years and 0 => p/t died before or at 3 years
            test_labels = np.where(test_surv_times > 3 * 365, 1, 0)
            aucs.append(roc_auc_score(y_true=test_labels, y_score=test_scores))
        mlp_no_fs_evaluations[cancer] = aucs
        # endregion MLP without SAGPool-based feature selection

        # region MLP with SAGPool-based feature selection
        files = [sorted([os.path.join(mlp_ex_dir, f) for f in os.listdir(mlp_ex_dir) if re.search(rx, f)]) for
                 rx in fs_rx]
        mlp_fs_files[cancer] = dict(zip(field_names, files))
        mlp_fs_data[cancer] = dict(zip(
            field_names,
            [[np.genfromtxt(f, delimiter=",") for f in files] for files in mlp_fs_files[cancer].values()]
        ))
        aucs = []
        for k in range(n_folds):
            # Exclude patients that were censored before 3 years
            test_surv_times = mlp_fs_data[cancer]["test_surv_times"][k]
            test_censor_bits = mlp_fs_data[cancer]["test_censor_bits"][k]
            test_curves = mlp_fs_data[cancer]["test_curves"][k]
            sel = test_surv_times >= 3 * 365
            test_censor_bits[sel] = 1.
            sel = test_censor_bits == 1.
            test_surv_times = test_surv_times[sel]
            test_scores = test_curves[sel, 1]  # Probability of surviving >= 3* 365 days
            # Need true labels where 1 => p/t died after 3 years and 0 => p/t died before or at 3 years
            test_labels = np.where(test_surv_times > 3 * 365, 1, 0)
            aucs.append(roc_auc_score(y_true=test_labels, y_score=test_scores))
        mlp_fs_evaluations[cancer] = aucs
        # endregion MLP with SAGPool-based feature selection

        # region GNN with SAGPool-based feature selection
        files = [sorted([os.path.join(gnn_ex_dir, f) for f in os.listdir(gnn_ex_dir) if re.search(rx, f)]) for
                 rx in fs_rx]
        gnn_fs_files[cancer] = dict(zip(field_names, files))
        gnn_fs_data[cancer] = dict(zip(
            field_names,
            [[np.genfromtxt(f, delimiter=",") for f in files] for files in gnn_fs_files[cancer].values()]
        ))
        aucs = []
        for k in range(n_folds):
            # Exclude patients that were censored before 3 years
            test_surv_times = gnn_fs_data[cancer]["test_surv_times"][k]
            test_censor_bits = gnn_fs_data[cancer]["test_censor_bits"][k]
            test_curves = gnn_fs_data[cancer]["test_curves"][k]
            sel = test_surv_times >= 3 * 365
            test_censor_bits[sel] = 1.
            sel = test_censor_bits == 1.
            test_surv_times = test_surv_times[sel]
            test_scores = test_curves[sel, 1]  # Probability of surviving >= 3* 365 days
            # Need true labels where 1 => p/t died after 3 years and 0 => p/t died before or at 3 years
            test_labels = np.where(test_surv_times > 3 * 365, 1, 0)
            aucs.append(roc_auc_score(y_true=test_labels, y_score=test_scores))
        gnn_fs_evaluations[cancer] = aucs
        # endregion GNN with SAGPool-based feature selection

    evaluations = dict(zip(model_types, [mlp_no_fs_evaluations, mlp_fs_evaluations, gnn_fs_evaluations]))

    # Create dicts of dataframes of performance metrics for each cancer type. Each dataframe has cancer types in rows
    # and CV test fold results in columns.
    auc_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    auc_means_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    auc_sems_dict = dict(zip(model_types, [dict(), dict(), dict()]))

    auc = dict()

    # Pretty slow
    for model in model_types:

        for k in range(n_folds):
            auc_dict[model][f"fold {k + 1}"] = dict()
            for cancer in cancer_types:
                auc_dict[model][f"fold {k + 1}"][cancer] = evaluations[model][cancer][k]

        auc[model] = pd.DataFrame(data=auc_dict[model]).round(decimals=2).transpose()

    for model in model_types:  # Loop above needs to complete before this loop can be run
        auc_means_dict[model] = auc[model].mean(axis=0)
        auc_sems_dict[model] = auc[model].sem(axis=0)

    auc_means = pd.DataFrame(data=auc_means_dict)
    auc_sems = pd.DataFrame(data=auc_sems_dict)

    mlp_shared_segment = f"mlp_{database}_{direction}_feature-selection="
    gnn_shared_segment = f"gnn_{database}_{merge}_{direction}_{graph_type}_feature-selection="

    file_name = os.path.join(output_dir, f"{mlp_shared_segment}no-sagpool_binary-auc.csv")
    auc["MLP - No feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{mlp_shared_segment}sagpool_binary-auc.csv")
    auc["MLP - SAGPool-based feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{gnn_shared_segment}sagpool_binary-auc.csv")
    auc["GNN - SAGPool"].to_csv(file_name)
