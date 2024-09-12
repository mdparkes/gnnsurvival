"""
Evaluate models from experiments where only a SAGPool version of the GNN was trained
"""
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
        help="The number of survival intervals modeled. If n_intervals is set to 2, 3 years will be used as the "
             "cutoff time.",
        type=int
    )
    args = vars(parser.parse_args())
    # endregion Parse args

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
    # endregion Directories

    n_folds = 5  # Does not include the tuning fold
    model_types = ["MLP - SAGPool-based feature selection",
                   "MLP - No feature selection",
                   "GNN - SAGPool"]

    # region Load data from disk
    mlp_fs_files = dict()
    mlp_fs_data = dict()
    mlp_fs_evaluations = dict()

    mlp_no_fs_files = dict()
    mlp_no_fs_data = dict()
    mlp_no_fs_evaluations = dict()

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

        # region MLP with SAGPool-based feature selection
        files = [sorted([os.path.join(mlp_ex_dir, f) for f in os.listdir(mlp_ex_dir) if re.search(rx, f)]) for
                 rx in fs_rx]
        mlp_fs_files[cancer] = dict(zip(field_names, files))
        data = [[np.genfromtxt(f, delimiter=",") for f in files] for files in mlp_fs_files[cancer].values()]
        mlp_fs_data[cancer] = dict(zip(field_names, data))
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

        # region MLP with no feature selection
        files = [sorted([os.path.join(mlp_ex_dir, f) for f in os.listdir(mlp_ex_dir) if re.search(rx, f)]) for
                 rx in no_fs_rx]
        mlp_no_fs_files[cancer] = dict(zip(field_names, files))
        data = [[np.genfromtxt(f, delimiter=",") for f in files] for files in mlp_no_fs_files[cancer].values()]
        mlp_no_fs_data[cancer] = dict(zip(field_names, data))
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
        # endregion MLP with no feature selection

        # region GNN with SAGPool-based feature selection
        files = [sorted([os.path.join(gnn_ex_dir, f) for f in os.listdir(gnn_ex_dir) if re.search(rx, f)]) for
                 rx in fs_rx]
        gnn_fs_files[cancer] = dict(zip(field_names, files))
        data = [[np.genfromtxt(f, delimiter=",") for f in files] for files in gnn_fs_files[cancer].values()]
        gnn_fs_data[cancer] = dict(zip(field_names, data))
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

    evaluations = dict(zip(model_types, [mlp_fs_evaluations, mlp_no_fs_evaluations, gnn_fs_evaluations]))

    # Create dicts of dataframes of performance metrics for each cancer type. Each dataframe has cancer types in rows
    # and CV test fold results in columns.
    concordance_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    concordance_means_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    concordance_sem_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    int_brier_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    int_brier_means_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    int_brier_sem_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    L1_loss_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    L1_loss_means_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    L1_loss_sem_dict = dict(zip(model_types, [dict(), dict(), dict()]))
    d_cal_dict = dict(zip(model_types, [dict(), dict(), dict()]))

    concordance = dict()
    int_brier = dict()
    L1_loss = dict()
    d_cal = dict()

    # Pretty slow
    for model in model_types:

        for k in range(n_folds):
            concordance_dict[model][f"fold {k + 1}"] = dict()
            int_brier_dict[model][f"fold {k + 1}"] = dict()
            L1_loss_dict[model][f"fold {k + 1}"] = dict()
            d_cal_dict[model][f"fold {k + 1}"] = dict()

            for cancer in cancer_types:
                vals = evaluations[model][cancer][k].concordance(ties="Risk")[0]
                concordance_dict[model][f"fold {k + 1}"][cancer] = vals

                vals = evaluations[model][cancer][k].integrated_brier_score()
                int_brier_dict[model][f"fold {k + 1}"][cancer] = vals

                vals = evaluations[model][cancer][k].l1_loss(method="Margin")
                # Exclude outlier L1 losses greater than 10,000
                vals = np.NaN if (vals > 10000) else vals
                L1_loss_dict[model][f"fold {k + 1}"][cancer] = vals

                vals = evaluations[model][cancer][k].d_calibration()[0]
                d_cal_dict[model][f"fold {k + 1}"][cancer] = vals

        concordance[model] = pd.DataFrame(data=concordance_dict[model]).round(decimals=2).transpose()
        int_brier[model] = pd.DataFrame(data=int_brier_dict[model]).round(decimals=2).transpose()
        L1_loss[model] = pd.DataFrame(data=L1_loss_dict[model]).round(decimals=2).transpose()
        d_cal[model] = pd.DataFrame(data=d_cal_dict[model]).round(decimals=2).transpose()

    for model in model_types:  # Loop above needs to complete before this loop can be run
        concordance_means_dict[model] = concordance[model].mean(axis=0)
        concordance_sem_dict[model] = concordance[model].sem(axis=0)

        int_brier_means_dict[model] = int_brier[model].mean(axis=0)
        int_brier_sem_dict[model] = int_brier[model].sem(axis=0)

        L1_loss_means_dict[model] = L1_loss[model].mean(axis=0)
        L1_loss_sem_dict[model] = L1_loss[model].sem(axis=0)

    concordance_means = pd.DataFrame(data=concordance_means_dict)
    concordance_sems = pd.DataFrame(data=concordance_sem_dict)
    int_brier_means = pd.DataFrame(data=int_brier_means_dict)
    int_brier_sems = pd.DataFrame(data=int_brier_sem_dict)
    L1_loss_means = pd.DataFrame(data=L1_loss_means_dict)
    L1_loss_sems = pd.DataFrame(data=L1_loss_sem_dict)

    mlp_shared_segment = f"mlp_{database}_{direction}_feature-selection="
    gnn_shared_segment = f"gnn_{database}_{merge}_{direction}_{graph_type}_feature-selection="

    file_name = os.path.join(output_dir, f"{mlp_shared_segment}sagpool_concordance.csv")
    concordance["MLP - SAGPool-based feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{mlp_shared_segment}no-sagpool_concordance.csv")
    concordance["MLP - No feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{gnn_shared_segment}sagpool_concordance.csv")
    concordance["GNN - SAGPool"].to_csv(file_name)

    file_name = os.path.join(output_dir, f"{mlp_shared_segment}sagpool_int_brier.csv")
    int_brier["MLP - SAGPool-based feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{mlp_shared_segment}no-sagpool_int_brier.csv")
    int_brier["MLP - No feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{gnn_shared_segment}sagpool_int_brier.csv")
    int_brier["GNN - SAGPool"].to_csv(file_name)

    file_name = os.path.join(output_dir, f"{mlp_shared_segment}sagpool_L1_loss.csv")
    L1_loss["MLP - SAGPool-based feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{mlp_shared_segment}no-sagpool_L1_loss.csv")
    L1_loss["MLP - No feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{gnn_shared_segment}sagpool_L1_loss.csv")
    L1_loss["GNN - SAGPool"].to_csv(file_name)

    file_name = os.path.join(output_dir, f"{mlp_shared_segment}sagpool_d_cal.csv")
    d_cal["MLP - SAGPool-based feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{mlp_shared_segment}no-sagpool_d_cal.csv")
    d_cal["MLP - No feature selection"].to_csv(file_name)
    file_name = os.path.join(output_dir, f"{gnn_shared_segment}sagpool_d_cal.csv")
    d_cal["GNN - SAGPool"].to_csv(file_name)

    # Plot version without L1 loss
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(11, 8)
    ax1.set_xlabel("Mean concordance")
    ax2.set_xlabel("Mean integrated Brier score")
    bar_height = 0.15
    lab_loc = np.arange(len(cancer_types))  # label locations

    multiplier = 0
    for model, means in concordance_means_dict.items():
        if multiplier % 4 == 0:
            color = "black"
        elif multiplier % 4 == 1:
            color = "darkgrey"
        elif multiplier % 4 == 2:
            color = "dimgrey"
        else:
            color = "whitesmoke"
        offset = bar_height * multiplier
        bars = ax1.barh(lab_loc + offset, means, bar_height,
                        color=color, edgecolor="black", xerr=concordance_sems[model], label=model)
        multiplier += 1

    multiplier = 0
    for model, means in int_brier_means_dict.items():
        if multiplier % 4 == 0:
            color = "black"
        elif multiplier % 4 == 1:
            color = "darkgrey"
        elif multiplier % 4 == 2:
            color = "dimgrey"
        else:
            color = "whitesmoke"
        offset = bar_height * multiplier
        bars = ax2.barh(lab_loc + offset, means, bar_height,
                        color=color, edgecolor="black", xerr=int_brier_sems[model], label=model)
        multiplier += 1

    ax1.set_yticks(lab_loc + bar_height, labels=cancer_types)
    ax2.set_yticks(lab_loc + bar_height, labels=None)

    ax2.yaxis.set_ticklabels([])

    fig.legend(loc="outside upper center", labels=model_types, ncols=3)
    plt.savefig(os.path.join(output_dir, f"final_model_performance_metrics_barplot_2panel.png"),
                bbox_inches="tight")