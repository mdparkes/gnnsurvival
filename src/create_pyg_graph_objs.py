"""
Script that creates PyTorch graph Data objects.
"""
import argparse
import numpy as np
import os
import pickle
import re
import torch

from keggpathwaygraphs import biopathgraph as bpg
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm
from typing import Dict, Tuple

# Local
from custom_data_types import NodeInfoDict, AdjacencyDict
from utilities import (filter_datasets_and_graph_info, list_source_and_target_indices,
                       load_expression_and_clinical_data, make_assignment_matrix, map_indices_to_names)


def unmerge_kegg_pathway_graph_info(
        node_info: NodeInfoDict, edge_info: AdjacencyDict, pathway_info: NodeInfoDict, min_nodes: int = 15
) -> Dict[str, Tuple[NodeInfoDict, AdjacencyDict]]:
    """
    Given a NodeInfoDict and AdjacencyDict for a graph that includes genes in all KEGG pathways, return a list of
    (NodeInfoDict, AdjacencyDict) tuples that represent each individual KEGG pathway. This function also filters out
    pathways that don't have at least `min_nodes`.

    :param node_info: A NodeInfoDict that contains node info from more than one KEGG pathway.
    :param edge_info: An AdjancencyDict that contains egde info corresponding to the nodes in `node_info`
    :param pathway_info: A NodeInfoDict for the KEGG pathway nodes, i.e. BRITE level C, whose children are nodes in
    `node_info`.
    :param min_nodes: The minimum number of nodes a KEGG pathway must have in order to be included in the output.
    :return: A dict of (NodeInfoDict, AdjacencyDict) tuples keyed by the KEGG pathways they represent, provided the
    KEGG pathways have at least `min_nodes` nodes.
    """
    output_dict = dict()
    for path, info in pathway_info.items():
        children = info["children"]
        if len(children) < min_nodes:
            # Skip pathway graphs with node sets smaller than min_nodes
            continue
        # Create new NodeInfoDict, AdjacencyDict objects for the pathway
        pathway_node_info = dict()
        pathway_edge_info = dict()
        for child in children:
            # Add nodes to the NodeInfoDict if they participate in the pathway
            pathway_node_info[child] = node_info[child]
            # Create a new edge set that includes edges whose sources and targets are members of the pathway
            if child in edge_info.keys():  # The node is an edge source node in edge_info
                pathway_edge_info[child] = set()
                edge_targets = edge_info[child]
                for target in edge_targets:
                    # Only include edge targets that participate in the pathway
                    if target[0] in children:
                        pathway_edge_info[child].add(target)
        # Add the pathway graph info to the output dict
        output_dict[path] = (pathway_node_info, pathway_edge_info)
    return output_dict


def make_edge_set_tensor(
        edge_set: str,
        adjacency_dict: AdjacencyDict,
        index_name_map: Dict[int, str],
        is_relational: bool) -> Tensor:
    """
    Creates an edge set Tensor for the specified edge set.

    :param edge_set: A string corresponding to one of the edge types in adjacency_dict
    :param adjacency_dict: An AdjacencyDict object listing target nodes and edge types for each source node
    :param index_name_map: A mapping of node indices in the GraphTensor node set keys/vals in adjacency_dict
    :param is_relational: True if graph is relational (multiple edge sets), False if graph is non-relational (single
    edge set)
    :return: An edge set Tensor for a specific edge set of a GraphTensor
    """
    edge_set_adjacency_dict = dict()
    # For each node in the graph, get targets connected to the source node by the specified edge type
    # Each key (source) is a GeneName, and each value (target set) is a Set[Tuple[GeneName, Tuple[EdgeTypeName]]
    # If the graph is non-relational (single edge set), ignore the edge types
    if is_relational:  # Relational graph with multiple edge sets
        for source, targets in adjacency_dict.items():
            edge_set_targets = set(t[0] for t in targets if edge_set in t[1])
            if len(edge_set_targets) > 0:
                edge_set_adjacency_dict[source] = edge_set_targets
    else:  # Non-relational graph with a single edge set
        for source, targets in adjacency_dict.items():
            edge_set_targets = set(t[0] for t in targets)
            if len(edge_set_targets) > 0:
                edge_set_adjacency_dict[source] = edge_set_targets

    source_indices, target_indices = list_source_and_target_indices(edge_set_adjacency_dict, index_name_map)
    edge_tns = torch.tensor([source_indices, target_indices], dtype=torch.long)
    return edge_tns


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
        help="Create a relational graph with one edge set per gene interaction type. Relational graphs use all "
             "interaction types except \"relation,\" which signifies any type of interaction",
        action="store_true"
    )
    parser.add_argument(
        "--nonrelational",
        help="Create a non-relational graph with one edge set.",
        dest="relational",
        action="store_false"
    )
    parser.add_argument(
        "--merge_pathways",
        help="If set, the script will merge all Reactome pathway graphs into a single large graph per patient. If not"
             "set, the default behavior is to create separate Reactome pathway graph Data/HeteroData objects.",
        dest="merge_pathways", action="store_true"
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
    directed = False if not args["directed"] else True
    direction = "directed" if directed else "undirected"
    relational = args["relational"]  # Boolean
    graph_type = "relational" if args["relational"] else "nonrelational"
    merge_pathways = False if not args["merge_pathways"] else True
    merge = "merged" if merge_pathways else "unmerged"
    cancer_types_to_examine = args["cancer_types"]
    k_folds = args["kfolds"]

    if reactome:
        graph_info_file = f"data/{database}_graph_{direction}.pkl"
    if brite:  # KEGG
        graph_info_file = f"data/{database}_graph.pkl"

    if len(cancer_types_to_examine) == 0:
        raise ValueError("A valid TCGA letter code must be supplied for at least one cancer type"
                         "as argument(s) to \"cancer_types\"")
    else:
        cancer_types_to_examine = [cancer_type.upper() for cancer_type in cancer_types_to_examine]

    # region Directories
    data_dir = args["output_dir"]
    # endregion Directories

    # region Files to read
    exprs_file = args["exprs_file"]
    clin_file = args["clin_file"]
    # endregion Files to read

    # region Files to write
    feature_names_files = []  # Files that contain KEGG IDs of genes used in a cancer type's graphs
    pathway_names_files = []  # Files that contain the pathway IDs represented by a cancer type's graphs
    assignment_matrix_files = [] if merge_pathways else None
    for cancer_type in cancer_types_to_examine:
        if merge_pathways:
            file_name = f"{cancer_type}_{database}_{direction}_pooling_assignment.pt"
            assignment_matrix_files.append(os.path.join(data_dir, cancer_type, file_name))
        feature_names_file_name = f"{cancer_type}_{database}_{merge}_{direction}_feature_names.pkl"
        feature_names_files.append(os.path.join(data_dir, cancer_type, feature_names_file_name))
        pathway_names_file_name = f"{cancer_type}_{database}_{merge}_{direction}_pathway_names.npy"
        pathway_names_files.append(os.path.join(data_dir, cancer_type, pathway_names_file_name))
    # endregion Files to write

    # region Check for necessary files and directories
    for cancer_type in cancer_types_to_examine:
        if not os.path.exists(os.path.join(data_dir, cancer_type)):
            os.makedirs(os.path.join(data_dir, cancer_type))

    if not os.path.exists(graph_info_file):
        raise FileNotFoundError("Run create_graph.py to parse graph structure from KEGG BRITE orthology, "
                                "or run create_reactome_graph.py to create Reactome graphs.")
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

    # Get node and edge info for the large graph that contains all pathways
    with open(graph_info_file, "rb") as file_in:
        graph_info = pickle.load(file_in)

    if reactome:
        # No name changes here, just using this dummy dict to avoid having different code for creating reactome graphs
        edge_set_names = {
            "Binding": "Binding",
            "Control_In_ACTIVATION_of_BiochemicalReaction": "Control_In_ACTIVATION_of_BiochemicalReaction",
            "Control_In_ACTIVATION_of_Degradation": "Control_In_ACTIVATION_of_Degradation",
            "Control_In_IHIBITION_of_BiochemicalReaction": "Control_In_IHIBITION_of_BiochemicalReaction",
            "Control_Out_ACTIVATION_of_BiochemicalReaction": "Control_Out_ACTIVATION_of_BiochemicalReaction",
            "Control_Out_ACTIVATION_of_TemplateReaction": "Control_Out_ACTIVATION_of_TemplateReaction",
            "Control_Out_INHIBITION_of_BiochemicalReaction": "Control_Out_INHIBITION_of_BiochemicalReaction",
            "Control_indirect": "Control_indirect",
            "Process_BiochemicalReaction": "Process_BiochemicalReaction"
        }
        if not merge_pathways:
            # Get a list of .pkl files containing node and edge info for each individual pathway
            node_info_dict_dir = os.path.join(data_dir, "Pathway", "pathways", "dicts")
            rx = re.compile(r".+_directed\.pkl") if directed else re.compile(r".+_undirected\.pkl")
            node_info_files = [f for f in sorted(os.listdir(node_info_dict_dir)) if rx.search(f)]
            node_info_files = [os.path.join(node_info_dict_dir, f) for f in node_info_files]

    if brite:
        edge_set_names = {
            "compound": "compound",
            "hidden compound": "hidden_compound",
            "activation": "activation",
            "inhibition": "inhibition",
            "expression": "expression",
            "repression": "repression",
            "indirect effect": "indirect_effect",
            "state change": "state_change",
            "binding/association": "binding_association",
            "dissociation": "dissociation",
            "missing interaction": "missing_interaction",
            "phosphorylation": "phosphorylation",
            "dephosphorylation": "dephosphorylation",
            "glycosylation": "glycosylation",
            "ubiquitination": "ubiquitination",
            "methylation": "methylation"
        }
        if not merge_pathways:
            path_info_dict = unmerge_kegg_pathway_graph_info(
                node_info=graph_info[3][0], edge_info=graph_info[3][1], pathway_info=graph_info[2][0], min_nodes=15
            )

    message = "Creating graph info objects"

    for i, cancer_type in enumerate(cancer_types_to_examine):
        print(f"Cancer {i + 1} of {len(cancer_types_to_examine)}: {cancer_type}")
        # region Create the graph information dictionaries
        """
        Some biopsies have NaN values for some genes. The feature set for a particular cancer type will exclude 
        genes with any NaN values among biopsies from that cancer type. Therefore, the node info dictionary for level D
        needs to be edited in a loop over cancer types.
        """
        sel_samples = clin_data.index[clin_data["acronym"] == cancer_type]  # Select obs. from single cancer type
        exprs_ss = exprs_data.loc[sel_samples, :].dropna(axis=1)  # Drop columns (genes) with missing values
        clin_ss = clin_data.loc[sel_samples, :]  # Match the clin_ss rows to the remaining rows in exprs_ss

        level_c_dict = graph_info[2][0].copy()

        tmp_data_dict = {}  # Accumulate graph objects for each patient

        if merge_pathways:  # Create one large graph per patient comprised of subgraphs for all pathways
            level_d_dict, level_d_adj = graph_info[3][0].copy(), graph_info[3][1].copy()

            exprs_ss, level_d_dict, level_d_adj, level_c_dict = filter_datasets_and_graph_info(
                exprs_ss, level_d_dict, level_d_adj, level_c_dict, min_nodes=15
            )

            # Write feature names used in final graph
            all_features = exprs_ss.columns.to_list()
            feature_names = [[feat for feat in all_features if feat in level_d_dict.keys()]]
            with open(feature_names_files[i], "wb") as file_out:
                pickle.dump(feature_names, file_out)

            # Write pathway names used in final graph
            pathway_names = list(level_c_dict.keys())
            np.save(pathway_names_files[i], np.array(pathway_names))

            # region Create graph coarsening assignment matrix
            level_c_dict = bpg.update_children(level_c_dict, set(level_d_dict.keys()))
            pooling_assignment = make_assignment_matrix(
                sender_graph_info=level_d_dict, sender_index_name_map=map_indices_to_names(level_d_dict),
                receiver_graph_info=level_c_dict, receiver_index_name_map=map_indices_to_names(level_c_dict)
            )
            torch.save(pooling_assignment, assignment_matrix_files[i])
            # endregion Create graph coarsening assignment matrix

            # region Create graph datasets
            age = clin_ss.loc[:, "age_at_initial_pathologic_diagnosis"]
            stage = clin_ss.loc[:, "pathologic_stage"]
            survival_times = clin_ss.loc[:, "days_to_last_known_alive"]
            censor_status = clin_ss.loc[:, "vital_status"]

            index_name_map = map_indices_to_names(level_d_dict)

            if relational:
                # region Create merged relational graphs
                # Create each edge index tensor only once -- same for all patients
                edge_indexes = dict()
                for original_name, formatted_name in edge_set_names.items():
                    edge_index_tensor = make_edge_set_tensor(original_name, level_d_adj, index_name_map, relational)
                    edge_indexes[formatted_name] = edge_index_tensor
                init_data = dict()
                for j, pt_id in tqdm(enumerate(exprs_ss.index), total=len(exprs_ss), desc=message):
                    # Add the edge sets to the graph data dict
                    for formatted_name, edge_index in edge_indexes.items():
                        init_data[("gene", formatted_name, "gene")] = {"edge_index": edge_index}
                    # Add node features and labels to the graph data dict
                    node_tensor = torch.reshape(torch.tensor(exprs_ss.iloc[j].to_list()), shape=(-1, 1))
                    init_data.update({"gene": {"x": node_tensor.float()}})  # Node features
                    init_data.update({"survival_time": torch.tensor(survival_times.iat[j], dtype=torch.float32),
                                      "deceased": torch.tensor(censor_status.iat[j], dtype=torch.uint8)})
                    graph_obj = HeteroData(init_data)
                    if pt_id not in tmp_data_dict.keys():
                        tmp_data_dict[pt_id] = [graph_obj]
                    else:
                        tmp_data_dict[pt_id].append(graph_obj)
                # endregion Create merged relational graphs

            else:
                # region Create merged nonrelational graphs
                edge_index_tensor = make_edge_set_tensor("relation", level_d_adj, index_name_map, relational)
                for j, pt_id in tqdm(enumerate(exprs_ss.index), total=len(exprs_ss), desc=message):
                    node_tensor = torch.reshape(torch.tensor(exprs_ss.iloc[j].to_list()), shape=(-1, 1))
                    graph_obj = Data(x=node_tensor.float(), edge_index=edge_index_tensor)
                    graph_obj["survival_time"] = torch.tensor(survival_times.iat[j], dtype=torch.float32)
                    graph_obj["deceased"] = torch.tensor(censor_status.iat[j], dtype=torch.uint8)
                    if pt_id not in tmp_data_dict.keys():
                        tmp_data_dict[pt_id] = [graph_obj]  # List consists of a single large graph object
                    else:
                        tmp_data_dict[pt_id].append(graph_obj)
                # endregion Create merged nonrelational graphs

            # endregion Create graph datasets

        else:  # Create one graph object per pathway and save as a collection per patient
            feature_names = []  # Updated with feature names as individual pathway graphs are processed
            pathway_names = []  # Updated with pathway names as they are read
            age = clin_ss.loc[:, "age_at_initial_pathologic_diagnosis"]
            stage = clin_ss.loc[:, "pathologic_stage"]
            survival_times = clin_ss.loc[:, "days_to_last_known_alive"]
            censor_status = clin_ss.loc[:, "vital_status"]

            if reactome:
                # Load individual pathway info dicts from disk
                rx = re.compile(fr"[^/]+(?=_{direction}\.pkl)")  # matches the name of the reactome pathway
                # node_info_files was sorted to guarantee predictable order over pathways
                for file in tqdm(node_info_files, total=len(node_info_files), desc=message):
                    with open(file, "rb") as file_in:
                        level_d_dict, level_d_adj = pickle.load(file_in)
                    path = rx.search(file)[0]
                    # create a small single-pathway dict for faster filtering
                    single_path_dict = {path: level_c_dict[path]}
                    path_exprs_ss, level_d_dict, level_d_adj, _ = filter_datasets_and_graph_info(
                        exprs_ss, level_d_dict, level_d_adj, single_path_dict, min_nodes=15
                    )
                    if len(level_d_dict) == 0:
                        # Skip graph creation for this pathway if it had fewer than 15 nodes with expression data
                        continue
                    all_features = path_exprs_ss.columns.to_list()  # All features in exprs_data
                    features_used = [feat for feat in all_features if feat in level_d_dict.keys()]
                    feature_names.append(features_used)  # Final features to use
                    # Record pathway name
                    pathway_names.append(path)

                    index_name_map = map_indices_to_names(level_d_dict)

                    if relational:

                        edge_indexes = dict()
                        for original_name, formatted_name in edge_set_names.items():
                            edge_index_tensor = make_edge_set_tensor(
                                original_name, level_d_adj, index_name_map, relational
                            )
                            edge_indexes[formatted_name] = edge_index_tensor
                        init_data = dict()
                        for j, pt_id in enumerate(path_exprs_ss.index):
                            # Add the edge sets to the graph data dict
                            for formatted_name, edge_index in edge_indexes.items():
                                init_data[("gene", formatted_name, "gene")] = {"edge_index": edge_index}
                            # Add node features and labels to the graph data dict
                            node_tensor = torch.reshape(torch.tensor(path_exprs_ss.iloc[j].to_list()), shape=(-1, 1))
                            init_data.update({"gene": {"x": node_tensor.float()}})  # Node features
                            init_data.update({"survival_time": torch.tensor(survival_times.iat[j], dtype=torch.float32),
                                              "deceased": torch.tensor(censor_status.iat[j], dtype=torch.uint8)})
                            graph_obj = HeteroData(init_data)
                            if pt_id not in tmp_data_dict.keys():
                                tmp_data_dict[pt_id] = [graph_obj]
                            else:
                                tmp_data_dict[pt_id].append(graph_obj)

                    else:

                        edge_index_tensor = make_edge_set_tensor("relation", level_d_adj, index_name_map, relational)
                        for j, pt_id in enumerate(path_exprs_ss.index):
                            node_tensor = torch.reshape(torch.tensor(path_exprs_ss.iloc[j].to_list()), shape=(-1, 1))
                            graph_obj = Data(x=node_tensor.float(), edge_index=edge_index_tensor)
                            graph_obj["survival_time"] = torch.tensor(survival_times.iat[j], dtype=torch.float32)
                            graph_obj["deceased"] = torch.tensor(censor_status.iat[j], dtype=torch.uint8)
                            if pt_id not in tmp_data_dict.keys():
                                tmp_data_dict[pt_id] = [graph_obj]
                            else:
                                tmp_data_dict[pt_id].append(graph_obj)

            if brite:
                # Instead of loading individual dicts from disk, use path_info_dict
                for path, info in tqdm(path_info_dict.items(), total=len(path_info_dict), desc=message):
                    # Create a small single pathway dict for faster filtering
                    single_path_dict = {path: level_c_dict[path]}
                    level_d_dict, level_d_adj = info
                    path_exprs_ss, level_d_dict, level_d_adj, _ = filter_datasets_and_graph_info(
                        exprs_ss, level_d_dict, level_d_adj, single_path_dict, min_nodes=15
                    )
                    if len(level_d_dict) == 0:
                        # Skip graph creation for this pathway if it had fewer than 15 nodes with expression data
                        continue
                    all_features = path_exprs_ss.columns.to_list()  # All features in exprs_data
                    features_used = [feat for feat in all_features if feat in level_d_dict.keys()]
                    feature_names.append(features_used)  # Final features to use
                    # Record pathway names
                    pathway_names.append(path)

                    index_name_map = map_indices_to_names(level_d_dict)

                    if relational:

                        edge_indexes = dict()
                        for original_name, formatted_name in edge_set_names.items():
                            edge_index_tensor = make_edge_set_tensor(
                                original_name, level_d_adj, index_name_map, relational
                            )
                            edge_indexes[formatted_name] = edge_index_tensor
                        init_data = dict()
                        for j, pt_id in enumerate(path_exprs_ss.index):
                            # Add the edge sets to the graph data dict
                            for formatted_name, edge_index in edge_indexes.items():
                                init_data[("gene", formatted_name, "gene")] = {"edge_index": edge_index}
                            # Add node features and labels to the graph data dict
                            node_tensor = torch.reshape(torch.tensor(path_exprs_ss.iloc[j].to_list()), shape=(-1, 1))
                            init_data.update({"gene": {"x": node_tensor.float()}})  # Node features
                            init_data.update({"survival_time": torch.tensor(survival_times.iat[j], dtype=torch.float32),
                                              "deceased": torch.tensor(censor_status.iat[j], dtype=torch.uint8)})
                            graph_obj = HeteroData(init_data)
                            if pt_id not in tmp_data_dict.keys():
                                tmp_data_dict[pt_id] = [graph_obj]
                            else:
                                tmp_data_dict[pt_id].append(graph_obj)

                    else:

                        edge_index_tensor = make_edge_set_tensor("relation", level_d_adj, index_name_map, relational)
                        for j, pt_id in enumerate(path_exprs_ss.index):
                            node_tensor = torch.reshape(torch.tensor(path_exprs_ss.iloc[j].to_list()), shape=(-1, 1))
                            graph_obj = Data(x=node_tensor.float(), edge_index=edge_index_tensor)
                            graph_obj["survival_time"] = torch.tensor(survival_times.iat[j], dtype=torch.float32)
                            graph_obj["deceased"] = torch.tensor(censor_status.iat[j], dtype=torch.uint8)
                            if pt_id not in tmp_data_dict.keys():
                                tmp_data_dict[pt_id] = [graph_obj]
                            else:
                                tmp_data_dict[pt_id].append(graph_obj)

            with open(feature_names_files[i], "wb") as file_out:
                pickle.dump(feature_names, file_out)

            pathway_names = np.array(pathway_names)
            np.save(pathway_names_files[i], pathway_names)

        print("Writing graphs to disk", end="... ", flush=True)
        output_dir = os.path.join(data_dir, cancer_type, "graphs", database, merge, direction, graph_type, "raw")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for j, (pt_id, graphs) in enumerate(tmp_data_dict.items()):
            age_tensor = torch.tensor(age.iat[j], dtype=torch.float32)
            stage_tensor = torch.tensor(stage.iat[j], dtype=torch.float32)
            time_tensor = torch.tensor(survival_times.iat[j], dtype=torch.float32)
            event_tensor = torch.tensor(censor_status.iat[j], dtype=torch.uint8)

            feature_data = (graphs, age_tensor, stage_tensor)
            label_data = (time_tensor, event_tensor)

            file_out = os.path.join(output_dir, f"{pt_id}.pt")
            torch.save(obj=(feature_data, label_data), f=file_out)
        print("Done")

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
