import argparse
import os
import numpy as np
import pandas as pd
import re
import torch

from matplotlib import pyplot as plt
from torchmtlr.utils import make_time_bins
# Local
from surveval import SurvivalEvaluator

if __name__ == "__main__":

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Evaluate models from an experiment"
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
    parser.add_argument(
        "-db", "--database",
        help="The database the graphs are constructed from. Takes \"reactome\" or \"kegg\".",
    )
    parser.add_argument(
        "--directed",
        help="If using reactome graphs, use this flag to use models over directed graphs. The default behavior is to "
             "use reactome graphs with both directed and undirected edges. No effect if using BRITE graphs. Note that"
             "the MLP inputs are affected by the decision to use undirected edges.",
        action="store_true"
    )
    parser.add_argument(
        "--relational",
        help="If set, use GNN models over relational graphs with multiple edge sets.",
        action="store_true"
    )
    parser.add_argument(
        "--nonrelational",
        help="If set, use GNN models over non-relational graphs with a single edge set.",
        dest="relational",
        action="store_false"
    )
    parser.add_argument(
        "--merge_pathways",
        help="If set, use models whose inputs consisted of a single conglomerate of merged pathway subgraphs. If not "
             "set, use models whose inputs are individual pathway graphs.",
        dest="merge_pathways", action="store_true"
    )
    parser.add_argument(
        "-i", "--n_intervals",
        help="The number of intervals to split time into for MTLR survival modeling. Intervals are determined from "
             "the n quantiles of survival times of uncensored patients. If n_intervals is set to 2, survival modeling "
             "equates to predicting whether a patient will survive at least until the single time point that separates "
             "the two intervals. If n_intervals is set to 2, 3 years will be used as the cutoff time.",
        type=int
    )
    args = vars(parser.parse_args())
    # endregion Parse args

    data_dir = args["data_dir"]
    output_dir = args["output_dir"]
    cancer_types = args["cancer_types"]
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    merge_pathways = True if args["merge_pathways"] else False
    merge = "merged" if merge_pathways else "unmerged"
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
    graph_type = "relational" if args["relational"] else "nonrelational"
    relational = True if graph_type == "relational" else False
    n_intervals = args["n_intervals"]
    n_folds = 5  # Does not include the tuning fold
    model_types = ["PGDNN - No feature selection",
                   "PGDNN - Feature selection",
                   "PathGNN - No feature selection",
                   "PathGNN - Feature selection"]

    # region Load data from disk
    mlp_no_fs_files = dict()
    mlp_no_fs_data = dict()
    mlp_no_fs_evaluations = dict()

    mlp_fs_files = dict()
    mlp_fs_data = dict()
    mlp_fs_evaluations = dict()

    gnn_no_fs_files = dict()
    gnn_no_fs_data = dict()
    gnn_no_fs_evaluations = dict()

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

    stats = pd.read_csv("experiment1/concordance_unpaired_two-sided_welch_pvals.csv", index_col=0)

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
        eval_objs = list()
        for k in range(n_folds):
            train_surv_times = mlp_no_fs_data[cancer]["train_surv_times"][k]
            train_censor_status = mlp_no_fs_data[cancer]["train_censor_bits"][k]
            if n_intervals == 2:  # Single time survival at 3 years
                intervals = torch.tensor([3 * 365.])
            else:
                # Only examples where censor_status == 1 will be used to calculate the time intervals
                intervals = make_time_bins(train_surv_times, num_bins=n_intervals, use_quantiles=True,
                                           event=train_censor_status)
            intervals = intervals.numpy()
            eval_objs.append(SurvivalEvaluator(
                predicted_survival_curves=mlp_no_fs_data[cancer]["test_curves"][k],  # (n_samples, n_time_points)
                time_coordinates=np.concatenate(([0.], intervals), axis=0),
                test_event_times=mlp_no_fs_data[cancer]["test_surv_times"][k],
                test_event_indicators=mlp_no_fs_data[cancer]["test_censor_bits"][k],
                train_event_times=mlp_no_fs_data[cancer]["train_surv_times"][k],
                train_event_indicators=mlp_no_fs_data[cancer]["train_censor_bits"][k],
                predict_time_method="Median"
            ))
        mlp_no_fs_evaluations[cancer] = eval_objs
        # endregion MLP without SAGPool-based feature selection

        # region MLP with SAGPool-based feature selection
        files = [sorted([os.path.join(mlp_ex_dir, f) for f in os.listdir(mlp_ex_dir) if re.search(rx, f)]) for
                 rx in fs_rx]
        mlp_fs_files[cancer] = dict(zip(field_names, files))
        mlp_fs_data[cancer] = dict(zip(
            field_names,
            [[np.genfromtxt(f, delimiter=",") for f in files] for files in mlp_fs_files[cancer].values()]
        ))
        eval_objs = list()
        for k in range(n_folds):
            train_surv_times = mlp_fs_data[cancer]["train_surv_times"][k]
            train_censor_status = mlp_fs_data[cancer]["train_censor_bits"][k]
            if n_intervals == 2:  # Single time survival at 3 years
                intervals = torch.tensor([3 * 365.])
            else:
                # Only examples where censor_status == 1 will be used to calculate the time intervals
                intervals = make_time_bins(train_surv_times, num_bins=n_intervals, use_quantiles=True,
                                           event=train_censor_status)
            intervals = intervals.numpy()
            eval_objs.append(SurvivalEvaluator(
                predicted_survival_curves=mlp_fs_data[cancer]["test_curves"][k],  # (n_samples, n_time_points)
                time_coordinates=np.concatenate(([0.], intervals), axis=0),
                test_event_times=mlp_fs_data[cancer]["test_surv_times"][k],
                test_event_indicators=mlp_fs_data[cancer]["test_censor_bits"][k],
                train_event_times=mlp_fs_data[cancer]["train_surv_times"][k],
                train_event_indicators=mlp_fs_data[cancer]["train_censor_bits"][k],
                predict_time_method="Median"
            ))
        mlp_fs_evaluations[cancer] = eval_objs
        # endregion MLP with SAGPool-based feature selection

        # region GNN without SAGPool-based feature selection
        files = [sorted([os.path.join(gnn_ex_dir, f) for f in os.listdir(gnn_ex_dir) if re.search(rx, f)]) for
                 rx in no_fs_rx]
        gnn_no_fs_files[cancer] = dict(zip(field_names, files))
        gnn_no_fs_data[cancer] = dict(zip(
            field_names,
            [[np.genfromtxt(f, delimiter=",") for f in files] for files in gnn_no_fs_files[cancer].values()]
        ))
        eval_objs = list()
        for k in range(n_folds):
            train_surv_times = gnn_no_fs_data[cancer]["train_surv_times"][k]
            train_censor_status = gnn_no_fs_data[cancer]["train_censor_bits"][k]
            if n_intervals == 2:  # Single time survival at 3 years
                intervals = torch.tensor([3 * 365.])
            else:
                # Only examples where censor_status == 1 will be used to calculate the time intervals
                intervals = make_time_bins(train_surv_times, num_bins=n_intervals, use_quantiles=True,
                                           event=train_censor_status)
            intervals = intervals.numpy()
            eval_objs.append(SurvivalEvaluator(
                predicted_survival_curves=gnn_no_fs_data[cancer]["test_curves"][k],  # (n_samples, n_time_points)
                time_coordinates=np.concatenate(([0.], intervals), axis=0),
                test_event_times=gnn_no_fs_data[cancer]["test_surv_times"][k],
                test_event_indicators=gnn_no_fs_data[cancer]["test_censor_bits"][k],
                train_event_times=gnn_no_fs_data[cancer]["train_surv_times"][k],
                train_event_indicators=gnn_no_fs_data[cancer]["train_censor_bits"][k],
                predict_time_method="Median"
            ))
        gnn_no_fs_evaluations[cancer] = eval_objs
        # endregion GNN without SAGPool-based feature selection

        # region GNN with SAGPool-based feature selection
        files = [sorted([os.path.join(gnn_ex_dir, f) for f in os.listdir(gnn_ex_dir) if re.search(rx, f)]) for
                 rx in fs_rx]
        gnn_fs_files[cancer] = dict(zip(field_names, files))
        gnn_fs_data[cancer] = dict(zip(
            field_names,
            [[np.genfromtxt(f, delimiter=",") for f in files] for files in gnn_fs_files[cancer].values()]
        ))
        eval_objs = list()
        for k in range(n_folds):
            train_surv_times = gnn_fs_data[cancer]["train_surv_times"][k]
            train_censor_status = gnn_fs_data[cancer]["train_censor_bits"][k]
            if n_intervals == 2:  # Single time survival at 3 years
                intervals = torch.tensor([3 * 365.])
            else:
                # Only examples where censor_status == 1 will be used to calculate the time intervals
                intervals = make_time_bins(train_surv_times, num_bins=n_intervals, use_quantiles=True,
                                           event=train_censor_status)
            intervals = intervals.numpy()
            eval_objs.append(SurvivalEvaluator(
                predicted_survival_curves=gnn_fs_data[cancer]["test_curves"][k],  # (n_samples, n_time_points)
                time_coordinates=np.concatenate(([0.], intervals), axis=0),
                test_event_times=gnn_fs_data[cancer]["test_surv_times"][k],
                test_event_indicators=gnn_fs_data[cancer]["test_censor_bits"][k],
                train_event_times=gnn_fs_data[cancer]["train_surv_times"][k],
                train_event_indicators=gnn_fs_data[cancer]["train_censor_bits"][k],
                predict_time_method="Median"
            ))
        gnn_fs_evaluations[cancer] = eval_objs
        # endregion GNN with SAGPool-based feature selection

    evaluations = dict(zip(model_types,
                           [mlp_no_fs_evaluations, mlp_fs_evaluations, gnn_no_fs_evaluations, gnn_fs_evaluations]))

    # Create dicts of dataframes of performance metrics for each cancer type. Each dataframe has cancer types in rows
    # and CV test fold results in columns.
    concordance_dict = dict(zip(model_types, [dict(), dict(), dict(), dict()]))
    concordance_means_dict = dict(zip(model_types, [dict(), dict(), dict(), dict()]))
    concordance_sem_dict = dict(zip(model_types, [dict(), dict(), dict(), dict()]))

    concordance = dict()

    # Pretty slow
    for model in model_types:

        for k in range(n_folds):
            concordance_dict[model][f"fold {k + 1}"] = dict()
            for cancer in cancer_types:
                vals = evaluations[model][cancer][k].concordance()[0]
                concordance_dict[model][f"fold {k + 1}"][cancer] = vals

        concordance[model] = pd.DataFrame(data=concordance_dict[model]).round(decimals=2).transpose()

    for model in model_types:  # Loop above needs to complete before this loop can be run
        concordance_means_dict[model] = concordance[model].mean(axis=0)
        concordance_sem_dict[model] = concordance[model].sem(axis=0)

    concordance_means = pd.DataFrame(data=concordance_means_dict)
    concordance_sems = pd.DataFrame(data=concordance_sem_dict)

    # Compare final models
    # Two-panel plot. Left panel is GNN vs MLP with feature selection. Right panel is GNN vs MLP with feature selection.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(11, 8)
    ax1.set_xlabel("Mean concordance")
    ax2.set_xlabel("Mean concordance")
    bar_height = 0.25
    lab_loc = np.arange(len(cancer_types))  # label locations

    # Left panel
    pvals = stats.loc["sagpool"]
    means_pgdnn1 = concordance_means_dict["PGDNN - Feature selection"]
    error = concordance_sems["PGDNN - Feature selection"]
    bar1_pgdnn = ax1.barh(lab_loc, means_pgdnn1, bar_height, xerr=error,
                          color="whitesmoke", edgecolor="dimgrey", hatch="/",
                          label="PGDNN - Feature selection")
    means_pathgnn1 = concordance_means_dict["PathGNN - Feature selection"]
    error = concordance_sems["PathGNN - Feature selection"]
    bar1_pathgnn = ax1.barh(lab_loc + bar_height, means_pathgnn1, bar_height, xerr=error,
                            color="darkgrey", edgecolor="dimgrey",
                            label="PathGNN - Feature selection")
    for i, cancer in enumerate(cancer_types):
        if pvals[cancer] < 0.05:
            coords = (means_pgdnn1[cancer] + 0.06, lab_loc[i] + bar_height / 2)
            ax1.annotate(text="\u2020", xy=coords, color="black", size=18, weight="bold", ha="center", va="center")

    # Right panel
    pvals = stats.loc["no-sagpool"]
    means_pgdnn2 = concordance_means_dict["PGDNN - No feature selection"]
    error = concordance_sems["PGDNN - No feature selection"]
    bar2_pgdnn = ax2.barh(lab_loc, means_pgdnn2, bar_height, xerr=error,
                          color="whitesmoke", edgecolor="dimgrey", hatch="/",
                          label="PGDNN - No feature selection")
    means_pathgnn2 = concordance_means_dict["PathGNN - No feature selection"]
    error = concordance_sems["PathGNN - No feature selection"]
    bar2_pathgnn = ax2.barh(lab_loc + bar_height, means_pathgnn2, bar_height, xerr=error,
                            color="darkgrey", edgecolor="dimgrey",
                            label="PathGNN - No feature selection")
    for i, cancer in enumerate(cancer_types):
        if pvals[cancer] < 0.05:
            coords = (means_pgdnn2[cancer] + 0.06, lab_loc[i] + bar_height / 2)
            ax2.annotate(text="\u2020", xy=coords, color="black", size=18, weight="bold", ha="center", va="center")

    ax1.set_axisbelow(True)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_yticks(lab_loc + bar_height / 2, labels=cancer_types)
    ax1.xaxis.grid()
    ax1.annotate(text="A", xy=(0.9, 11), size=32, weight="bold", ha="center", va="center")

    ax2.set_axisbelow(True)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_yticks(lab_loc + bar_height / 2, labels=None)
    ax2.yaxis.set_ticklabels([])
    ax2.xaxis.grid()
    ax2.annotate(text="B", xy=(0.9, 11), size=32, weight="bold", ha="center", va="center")

    fig.legend(handles=[bar2_pgdnn, bar2_pathgnn], labels=["PGDNN", "PathGNN"], loc="outside upper center", ncols=2)

    plt.savefig(os.path.join(output_dir, f"manuscript_concordance_barplots.png"), bbox_inches="tight")
