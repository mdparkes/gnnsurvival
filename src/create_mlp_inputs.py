"""
Write MLP surival model data to disk. The create_pyg_graph_objs.py script should be run before running this script
because this script needs to restrict the input tensors to features that are used in the inputs to GNN survival models.
"""
import argparse
import numpy as np
import os
import pickle
import torch

from keggpathwaygraphs import biopathgraph as bpg
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Local
from utilities import (filter_datasets_and_graph_info, load_expression_and_clinical_data, make_assignment_matrix,
                       map_indices_to_names)


def main():

    # region Parse command line args
    parser = argparse.ArgumentParser(
        description="Creates PyG Data objects (graphs) from biopsy RNAseq data and writes them to disk."
    )
    parser.add_argument(
        "-e", "--exprs_file",
        help="The path to the csv file containing RNA expression data",
        default="data/tcga_exprs.csv"

    )
    parser.add_argument(
        "-c", "--clin_file",
        help="The path to the csv file containing the clinical data",
        default="data/tcga_clin.csv"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="The path to the directory where the Data objects will be written. For each cancer type passed as an "
             "argument to --cancer_types, graph Data objects will be written to ./[output_dir]/[cancer_type]/graphs.",
        type=str
    )
    parser.add_argument(
        "-t", "--cancer_types",
        help="SCANB and/or TCGA letter codes of cancer types for which to construct individual graph Data objects",
        nargs="+",
        default=[]
    )
    parser.add_argument(
        "-db", "--database",
        help="The database the graphs are constructed from for the GNN survival model. Takes \"reactome\" or \"kegg\".",
    )
    parser.add_argument(
        "--directed",
        help="If using reactome graphs, use this flag to learn over directed graphs. The default behavior is to use "
             "reactome graphs with both directed and undirected edges. No effect if using KEGG/BRITE graphs.",
        action="store_true"
    )
    parser.add_argument(
        "-k", "--kfolds",
        help="The number of cross validation folds to use for testing model generalization",
        type=int,
        default=6
    )
    args = vars(parser.parse_args())
    # endregion Parse command line args

    # Use "brite" as database name instead of "kegg" to align with naming conventions in upstream code
    database = "reactome" if args["database"].lower() == "reactome" else "brite"
    reactome = True if database == "reactome" else False
    brite = True if database == "brite" else False
    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"
    cancer_types_to_examine = args["cancer_types"]
    k_folds = args["kfolds"]

    if len(cancer_types_to_examine) == 0:
        raise ValueError("Either \"SCANB\" or a valid TCGA letter code for at least one cancer type must be supplied "
                         "as argument(s) to \"cancer_types\"")
    else:
        cancer_types_to_examine = [cancer_type.upper() for cancer_type in cancer_types_to_examine]

    # region Directories
    data_dir = args["output_dir"]
    # endregion Directories

    # region Files to read
    exprs_file = args["exprs_file"]
    clin_file = args["clin_file"]
    graph_info_file = f"data/{database}_graph_{direction}.pkl" if reactome else f"data/{database}_graph.pkl"
    graph_feature_names_files = []
    for cancer_type in cancer_types_to_examine:
        file_name = f"{cancer_type}_{database}_unmerged_{direction}_feature_names.pkl"
        graph_feature_names_files.append(os.path.join(data_dir, cancer_type, file_name))
    # endregion Files to read

    # region Files to write
    assignment_matrix_files = []
    feature_names_files = []
    pathway_names_files = []
    for cancer_type in cancer_types_to_examine:
        file_name = f"{cancer_type}_{database}_{direction}_mlp_mask.pt"
        assignment_matrix_files.append(os.path.join(data_dir, cancer_type, file_name))
        file_name = f"{cancer_type}_{database}_{direction}_mlp_feature_names.pkl"
        feature_names_files.append(os.path.join(data_dir, cancer_type, file_name))
        file_name = f"{cancer_type}_{database}_{direction}_mlp_pathway_names.npy"
        pathway_names_files.append(os.path.join(data_dir, cancer_type, file_name))
    # endregion Files to write

    # region Check for necessary files and directories
    for cancer_type in cancer_types_to_examine:
        if not os.path.exists(os.path.join(data_dir, cancer_type)):
            os.makedirs(os.path.join(data_dir, cancer_type))
    # endregion Check for necessary files and directories

    # Load and process expression data and clinical data
    exprs_data, clin_data = load_expression_and_clinical_data(exprs_file, clin_file)

    # Prepare age, stage data per Liang et al in https://github.com/BioAI-kits/PathGNN/blob/master/data.py
    replacements = {
        "pathologic_stage": {
            "Stage 0": 0.0, "Stage X": 0.0, np.NaN: 0.0,
            "Stage I": 0.2, "Stage IA": 0.2, "Stage IB": 0.2, "Stage IS": 0.2, "I or II NOS": 0.2,
            "Stage II": 0.4, "Stage IIA": 0.4, "Stage IIB": 0.4, "Stage IIC": 0.4,
            "Stage III": 0.6, "Stage IIIA": 0.6, "Stage IIIB": 0.6, "Stage IIIC": 0.6,
            "Stage IV": 0.8, "Stage IVA": 0.8, "Stage IVB": 0.8, "Stage IVC": 0.8,
        },
        "age_at_initial_pathologic_diagnosis": {
            np.NaN: 50.0
        }
    }
    clin_data = clin_data.replace(to_replace=replacements)
    clin_data["age_at_initial_pathologic_diagnosis"] = clin_data["age_at_initial_pathologic_diagnosis"] / 100.00


    # Load graph info dicts (used for gene and pathway filtering prior to MLP input tensor construction)
    with open(graph_info_file, "rb") as file_in:
        graph_info = pickle.load(file_in)

    for i, cancer_type in enumerate(cancer_types_to_examine):
        print(f"Cancer {i + 1} of {len(cancer_types_to_examine)}: {cancer_type}")

        # Get the features that are used for the GNNs. These will be used to pre-filter the expression data,
        # excluding any genes that are not used in the graph(s).
        with open(graph_feature_names_files[i], "rb") as file_in:
            gnn_feature_names = pickle.load(file_in)
        tmp_set = set()
        for pathway_gene_list in gnn_feature_names:
            tmp_set = tmp_set.union(set(pathway_gene_list))
        gnn_feature_names = list(tmp_set)
        del tmp_set

        # Make working copies of the graph data
        level_c_dict = graph_info[2][0].copy()
        level_d_dict, level_d_adj = graph_info[3][0].copy(), graph_info[3][1].copy()

        # Filter data to include only relevant features and observations
        sel_samples = clin_data.index[clin_data["acronym"] == cancer_type]  # Select obs. from single cancer type
        # Drop genes (columns) that have missing values or aren't used in the graph inputs to GNN(s)
        exprs_ss = exprs_data.loc[sel_samples, gnn_feature_names].dropna(axis=1)
        clin_ss = clin_data.loc[sel_samples, :]  # Match the clin_ss rows to the remaining rows in exprs_ss
        exprs_ss, level_d_dict, _, level_c_dict = filter_datasets_and_graph_info(
            exprs_ss, level_d_dict, level_d_adj, level_c_dict, min_nodes=15
        )

        # Write feature names in order of appearance in the MLP input tensors
        feature_names = [[feat for feat in exprs_ss.columns.to_list() if feat in level_d_dict.keys()]]
        with open(feature_names_files[i], "wb") as file_out:
            pickle.dump(feature_names, file_out)

        # Write pathway names that correspond to the units of the MLP's first hidden layer
        pathway_names = list(level_c_dict.keys())
        np.save(pathway_names_files[i], np.array(pathway_names))

        # region Create weight mask for sparse MLP
        level_c_dict = bpg.update_children(level_c_dict, set(level_d_dict.keys()))
        pooling_assignment = make_assignment_matrix(
            sender_graph_info=level_d_dict, sender_index_name_map=map_indices_to_names(level_d_dict),
            receiver_graph_info=level_c_dict, receiver_index_name_map=map_indices_to_names(level_c_dict)
        )
        torch.save(pooling_assignment, assignment_matrix_files[i])
        # endregion Create weight mask for sparse MLP

        # region Create input tensors
        age = clin_ss.loc[:, "age_at_initial_pathologic_diagnosis"]
        stage = clin_ss.loc[:, "pathologic_stage"]
        survival_times = clin_ss.loc[:, "days_to_last_known_alive"]
        censor_status = clin_ss.loc[:, "vital_status"]

        index_name_map = map_indices_to_names(level_d_dict)
        # endregion Create input tensors

        output_dir = os.path.join(data_dir, cancer_type, "mlp_inputs", database, direction, "raw")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for j, pt_id in tqdm(enumerate(exprs_ss.index), total=len(exprs_ss), desc="Writing MLP input data to disk"):

            exprs_tensor = torch.tensor(exprs_ss.iloc[j].to_list(), dtype=torch.float32)
            age_tensor = torch.tensor(age.iat[j], dtype=torch.float32)
            stage_tensor = torch.tensor(stage.iat[j], dtype=torch.float32)
            time_tensor = torch.tensor(survival_times.iat[j], dtype=torch.float32)
            event_tensor = torch.tensor(censor_status.iat[j], dtype=torch.uint8)

            feature_data = (exprs_tensor, age_tensor, stage_tensor)
            label_data = (time_tensor, event_tensor)

            file_out = os.path.join(output_dir, f"{pt_id}.pt")
            torch.save(obj=(feature_data, label_data), f=file_out)

        # region Create cross validation folds
        path_out = os.path.join(data_dir, cancer_type, "train_test_split_names.pkl")
        if not os.path.exists(path_out):
            print("Indexing training, validation, and test folds", end="... ", flush=True)
            cv_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=423)
            cv_split_names = []
            censor_status = censor_status.astype(np.bool_)
            for train_idx, test_idx in cv_splitter.split(exprs_ss, y=censor_status):
                # train_idx and test_idx are lists of indices
                train_names = [f"{name}.pt" for name in exprs_ss.iloc[train_idx].index]
                test_names = [f"{name}.pt" for name in exprs_ss.iloc[test_idx].index]
                cv_split_names.append((train_names, test_names))
            with open(path_out, "wb") as file_out:
                pickle.dump(cv_split_names, file_out)
            print("Done")


if __name__ == "__main__":
    main()
