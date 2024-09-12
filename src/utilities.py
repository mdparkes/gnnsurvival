"""
Portions of this code were adapted from survivalEVAL code by Shi-ang Qi (2021) under the MIT license:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import itertools
import numpy as np
import os
import pandas as pd
import re
import torch
import warnings

from dataclasses import dataclass, field
from rpy2 import robjects
from scipy import integrate
from torch import Tensor
from typing import Dict, List, Set, Tuple

# Local
from keggpathwaygraphs import biopathgraph as bpg

from custom_data_types import NodeInfoDict, AdjacencyDict, NumericArrayLike
from models import KaplanMeier


def maybe_create_directories(*dirs) -> None:
    """Check if dirs exist and creates them if necessary"""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def map_indices_to_names(nodes: NodeInfoDict) -> Dict[int, str]:
    """Create a dictionary of node names keyed by integer index values"""
    return dict(zip(range(len(nodes)), nodes.keys()))


def load_expression_and_clinical_data(exprs_file: str, clin_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading and processing gene expression and clinical data", end="... ", flush=True)
    exprs_data = pd.read_csv(exprs_file, index_col=0)
    clin_data = pd.read_csv(
        clin_file,
        encoding="windows-1252",
        index_col="aliquot_submitter_id",
        na_values=["[Not Applicable]", "[Completed]", "[Discrepancy]", "[Not Available]", "[Unknown]", "'--"],
        usecols=["aliquot_submitter_id", "bcr_patient_barcode", "acronym", "gender", "days_to_collection",
                 "vital_status", "days_to_death", "days_to_last_followup", "age_at_initial_pathologic_diagnosis",
                 "pathologic_stage"]
    )

    # Process clinical data
    clin_data = clin_data.loc[exprs_data.index, :]  # Make sure the order of the rows match
    clin_data["days_to_last_known_alive"] = clin_data.days_to_death.fillna(clin_data.days_to_last_followup)
    clin_data.loc[clin_data["vital_status"] == "Alive", "vital_status"] = 0
    clin_data.loc[clin_data["vital_status"] == "Dead", "vital_status"] = 1
    # GBM and STAD have NaN vital status. Record a 0 in vital status if days to last followup is known. If days to death
    # is known, record 1, even if days to followup was also given (we prefer 1 over 0)
    clin_data.loc[clin_data["vital_status"].isnull() & clin_data["days_to_last_followup"].notnull(), "vital_status"] = 0
    clin_data.loc[clin_data["vital_status"].isnull() & clin_data["days_to_death"].notnull(), "vital_status"] = 1
    # Exclude biopsies from patients with null vital status or negative event/censoring times
    sel_samples = (clin_data["vital_status"].notnull()) & (clin_data["days_to_last_known_alive"] >= 0.)
    clin_data = clin_data[sel_samples]

    # Process expression data
    exprs_data = exprs_data.loc[clin_data.index, :]  # Filter exprs_data according to what remains in clin_data
    # There are some samples with NaN values for certain genes; these will be dealt with later on a per-cancer basis
    exprs_data = np.log1p(exprs_data)
    print("Done", flush=True)

    # Save mappings of feature names to KEGG hsa IDs, ENTREZ IDs, and gene symbols
    print("Creating a mapping of feature names to KEGG hsa IDs, ENTREZ IDs, and gene symbols", end="... ", flush=True)
    orig_feat_names = exprs_data.columns
    reg_exp1 = re.compile(r"^[^|]+")  # Match gene symbol
    reg_exp2 = re.compile(r"(?<=\|)\d+")  # Match ENTREZ ID
    gene_symbols = [reg_exp1.search(string).group(0) for string in orig_feat_names]
    entrez_ids = [reg_exp2.search(string).group(0) for string in orig_feat_names]
    kegg_ids = [f"hsa{entrez}" for entrez in entrez_ids]
    feature_map = pd.DataFrame({
        "original": orig_feat_names,
        "entrez": entrez_ids,
        "symbol": gene_symbols,
        "kegg": kegg_ids
    })
    feature_map.to_csv("data/feature_map.csv", index_label=False)
    new_names = dict(zip(orig_feat_names, kegg_ids))
    exprs_data.rename(columns=new_names, inplace=True)  # Replace original names with KEGG IDs
    print("Done", flush=True)

    return exprs_data, clin_data


def filter_datasets_and_graph_info(
        exprs_df: pd.DataFrame, node_info: NodeInfoDict, edge_info: AdjacencyDict, pathway_info: NodeInfoDict,
        min_nodes: int = 15
) -> Tuple[pd.DataFrame, NodeInfoDict, AdjacencyDict, NodeInfoDict]:
    """
    Given a gene expression DataFrame, a NodeInfoDict and AdjacencyDict for a graph whose nodes represent genes in a
    pathway, a NodeInfoDict whose nodes represent the pathways themselves, and a minimum number of nodes per
    pathway, this function removes zero-expression genes (nodes) from the expression DataFrame and gene graph,
    and also genes that only appear in pathways with fewer than the specified minimum number of nodes. Those pathways
    are also removed from the pathway NodeInfoDict.

    When nodes are removed from `node_info`, edges that have the removed nodes as their source in edge_info are also
    removed. A new edge is formed between the target node and any node that had the removed node as its target. For
    example, if A -> B -> C <- D is a graph and node B is removed, the new graph becomes A -> C <- D.

    Note: When edge types are used in downstream applications, the edge removal may invalidate the graph. For
    example, consider A -[activates]-> B -[inhibits]-> C -[activates]-> D. Depending on the mechanism of inhibition
    of C by B, this may imply that A -[inhibits]-> D through the action of B, but if B and C are removed via remove_node
    the result would be A -[activates]-> D. This is not an issue when edge types are not considered in downstream
    applications.

    :param exprs_df: A DataFrame with gene expression counts
    :param node_info: A NodeInfoDict for a graph whose nodes represent genes
    :param edge_info: An AdjacencyDict for a graph whose nodes represent genes
    :param pathway_info: A NodeInfoDict for a graph whose nodes represent pathways
    :param min_nodes: The minimum number of nodes that a pathway must have in order to survive filtering
    :return: Returns the filtered input objects
    """
    # Remove genes from node_info and edge_info if they either don't appear in exprs_df or have zero expression
    # across all biopsies in exprs_df. Create edges between parents and children of the removed nodes in edge_info.
    for_removal = [gene for gene in node_info.keys() if gene not in exprs_df.columns]
    for gene in for_removal:
        node_info, edge_info = bpg.remove_node(gene, node_info, edge_info)
    node_info, edge_info = remove_zero_exprs(exprs_df, node_info, edge_info)
    # Restrict the members of each pathway to genes in node_info
    pathway_info = bpg.update_children(pathway_info, set(node_info.keys()))
    # Exclude pathways with < min_nodes nodes and create a set of nodes that appear in any of the remaining pathways
    # Returns empty dicts/DataFrame if all pathways are dropped
    new_pathway_info = dict()
    remaining_nodes = set()
    for path, info in pathway_info.items():
        children = info["children"]
        if len(children) < min_nodes:
            continue
        new_pathway_info[path] = info
        remaining_nodes = remaining_nodes.union(children)  # Update remaining nodes
    # Form a list of nodes that are not in any of the remaining pathways
    if len(remaining_nodes) > 0:
        for_removal = [gene for gene in node_info.keys() if gene not in remaining_nodes]
        for gene in for_removal:
            node_info, edge_info = bpg.remove_node(gene, node_info, edge_info)
        # Restrict exprs_df to genes that remain in node_info. exprs_df has its columns (genes) ordered to match the
        # order of the keys in node_info
        all_features = exprs_df.columns.to_numpy()  # All features in exprs_data; still contains zero-expression genes
        sel_features = [list(all_features).index(gene) for gene in node_info.keys()]  # non-zero exprs indices
        feature_names = all_features[sel_features]  # Final features to use
        exprs_df = exprs_df.loc[:, feature_names]  # Match the order of keys in node_info
    else:
        # There are no remaining nodes because all pathways were eliminated, so node_info, edge_info, and exprs_df
        # are emptied
        node_info = dict()
        edge_info = dict()
        exprs_df = exprs_df.loc[:, []]  # Empty DataFrame

    return exprs_df, node_info, edge_info, new_pathway_info


def remove_zero_exprs(
        data: pd.DataFrame, nodes: NodeInfoDict, adjacency: AdjacencyDict
) -> Tuple[NodeInfoDict, AdjacencyDict]:
    """Remove genes with no detectable counts in all biopsies"""
    sel_zero_exprs = data.sum(axis=1) == 0
    zero_exprs = data[sel_zero_exprs].index
    for target in zero_exprs:
        nodes, adjacency = bpg.remove_node(target, nodes, adjacency)
    return nodes, adjacency



def make_assignment_matrix(
    sender_graph_info: NodeInfoDict,
    receiver_graph_info: NodeInfoDict,
    sender_index_name_map: Dict[int, str],
    receiver_index_name_map: Dict[int, str]
) -> Tensor:
    """
    During graph coarsening, node features in one graph are aggregated to serve as the initial node features for a
    coarsened graph with fewer nodes. Supposing that the sender graph (the graph to be coarsened) and the receiver
    graph (the coarsened graph) have already been defined, `make_assignment_matrix` returns a matrix that specifies
    the mapping of information from nodes in the sender graph to nodes in the receiver graph. For example,
    given a sender graph with nodes {A, B, C, D} and a receiver graph with nodes {P, Q}, the assignment matrix might
    specify that the features from node P derive their values from the features of nodes A and B, and Q's features
    derive their values from the features of C and D.

    :param sender_graph_info: A dictionary that defines the sender graph
    :param receiver_graph_info: A dictionary that defines the receiver and its nodes' relationships to sender nodes
    :param sender_index_name_map: A mapping of numeric indices to node names for the sender graph
    :param receiver_index_name_map: A mapping of numeric indices to node names for the receiver graph
    :return: An assignment matrix in Tensor form
    """
    n_rows = len(receiver_graph_info)
    n_cols = len(sender_graph_info)

    sender_name_list = list(sender_index_name_map.values())
    receiver_name_list = list(receiver_index_name_map.values())

    assignment_matrix = np.zeros(shape=(n_rows, n_cols), dtype=float)

    for i in range(n_rows):
        r_name = receiver_name_list[i]
        s_names = list(receiver_graph_info[r_name]["children"])  # Names of nodes in sender to receive info from
        j = [sender_name_list.index(val) for val in s_names]  # Indices of nodes in sender to receive info from
        assignment_matrix[i, j] = 1

    assignment_matrix = torch.tensor(assignment_matrix, dtype=torch.float32)

    return assignment_matrix



def list_source_and_target_indices(
        adjacency_dict: Dict[str, Set[str]], index_map: Dict[int, str]
) -> Tuple[List[int], List[int]]:
    """
    Returns a tuple with two equal-length lists of node indices, denoting an edge between the ith items of the
    lists. Edges are directed; an undirected edge is represented as two directed edges.

    :param adjacency_dict: A dict that indicates genes that are connected by an edge in KEGG pathways. The keys are
    KEGG gene ID strings, and the values are sets of KEGG gene ID strings that the keyed gene is connected with via
    an edge.
    :param index_map: A mapping from numeric node indices (zero-indexed) to KEGG gene ID strings.
    :return: A tuple of equal-length lists of nodes that are connected by an edge
    """
    source_idx = list()
    target_idx = list()
    index_map_val_list = list(index_map.values())  # For finding the node index that corresponds to the KEGG ID
    for idx, name in index_map.items():
        if name in adjacency_dict.keys():  # Not all nodes have outgoing edges
            # Add the node index to the source list once per node it shares an edge with
            updated_neighbors = set(neigh for neigh in adjacency_dict[name] if neigh in index_map.values())
            adjacency_dict[name] = updated_neighbors  # Only retain neighbors that are actually in the subgraph
            source_idx.extend(itertools.repeat(idx, len(adjacency_dict[name])))
            targets_to_add = list()
            for neigh in adjacency_dict[name]:  # iterates over a `set` object
                targets_to_add.append(index_map_val_list.index(neigh))
            target_idx.extend(targets_to_add)
    return source_idx, target_idx


def make_subgraph_info_dicts(
        nodes_dict: NodeInfoDict, adjacency_dict: AdjacencyDict, nodes: Set[str]
) -> Tuple[NodeInfoDict, AdjacencyDict]:
    """
    This function takes dictionaries of node and adjacency (edge) information that define a graph and returns a
    tuple of new dictionaries that define a subgraph on said graph. The subgraph contains nodes in the set passed to
    the `nodes` parameter. It is expected that `nodes_dict` is a dictionary keyed by node names. This function
    does not update information about the children of each node if that is one of the information fields; the
    children will have to be updated separately they are to include only those nodes that are in a subgraph that
    is immediately subordinate to the one being created in a graph coarsening hierarchy.

    :param nodes_dict: A nested dictionary of information about each node in the graph
    :param adjacency_dict: A dictionary giving the set of neighboring nodes that receive edges from the central node
    :param nodes: The nodes to use in the subgraph
    :return: A tuple whose first item is the subgraph's node information dictionary and whose second item is the
    subgraph's adjacency dictionary.
    """
    # If the adjacency dict has edge type information, it must be supplied as a tuple whose first item is a string
    # giving the node name and whose second item is a tuple of edge types.
    adjacency_has_edge_types = all(
        [isinstance(neighbor, tuple) for neighbor_set in adjacency_dict.values() for neighbor in neighbor_set]
    )
    nodes_dict_new = {k: nodes_dict[k] for k in nodes}
    adjacency_dict_new = dict()
    for k in nodes_dict_new.keys():
        try:
            neighbor_set = adjacency_dict[k]
        except KeyError:  # Not all nodes have neighbors
            continue
        adjacency_dict_new[k] = set()
        # Restrict neighbors to subsets of nodes present in the subgraph
        if adjacency_has_edge_types:
            for neighbor in neighbor_set:
                if neighbor[0] in nodes_dict_new.keys():
                    adjacency_dict_new[k].add(neighbor)
        else:
            for neighbor in neighbor_set:
                if neighbor in nodes_dict_new.keys():
                    adjacency_dict_new[k].add(neighbor)

    return nodes_dict_new, adjacency_dict_new


def check_and_convert(*args):
    """ Makes sure that the given inputs are numpy arrays, list,
        tuple, panda Series, pandas DataFrames, or tensorflow Tensors.

        Also makes sure that the given inputs have the same shape.

        Then convert the inputs to numpy array.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    last_length = ()
    for i, arg in enumerate(args):

        if len(arg) == 0:
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)

        else:

            if isinstance(arg, np.ndarray):
                x = (arg.astype(np.double),)
            elif isinstance(arg, list):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, tuple):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, pd.Series):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, pd.DataFrame):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, torch.Tensor):
                x = (arg.numpy().astype(np.double),)
            else:
                error = """{arg} is not a valid data format. Only use 'list', 'tuple', 'np.ndarray', 'tf.Tensor', 
                        'pd.Series', 'pd.DataFrame'""".format(arg=type(arg))
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, \
                        f"Shapes between {i-1}-th input array and {i}-th input array are inconsistent"
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result


def predict_prob_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_time: float
) -> float:
    """
    Quote from ISDEvaluation/Evaluations/EvaluationHelperFunction.R
    We need some type of predict function for survival curves - here we build a spline to fit the survival model curve.
    This spline is the monotonic spline using the hyman filtering of the cubic Hermite spline method,
    see https://en.wikipedia.org/wiki/Monotone_cubic_interpolation. Also see help(splinefun).

    Note that we make an alteration to the method because if the last two time points
    have the same probability (y value) then the spline is constant outside the training time range.
    We need this to be a decreasing function outside the training data, so instead we take the linear fit of (0,1)
    and the last time point we have (p,t*) and then apply this linear function to all points outside our fit.
    """
    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    if target_time > max_time:
        # func: y = slope * x + 1, the minimum prob should be 0
        predict_probability = max(slope * target_time + 1, 0)
    else:
        predict_probability = np.array(spline(float(target_time))).item()

    return predict_probability


def predict_multi_probs_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_times: NumericArrayLike
) -> np.ndarray:
    """
    Quote from ISDEvaluation/Evaluations/EvaluationHelperFunction.R
    We need some type of predict function for survival curves - here we build a spline to fit the survival model curve.
    This spline is the monotonic spline using the Hyman filtering of the cubic Hermite spline method,
    see https://en.wikipedia.org/wiki/Monotone_cubic_interpolation. Also see help(splinefun).

    Note that we make an alteration to the method because if the last two time points
    have the same probability (y value) then the spline is constant outside the training time range.
    We need this to be a decreasing function outside the training data, so instead we take the linear fit of (0,1)
    and the last time point we have (p,t*) and then apply this linear function to all points outside our fit.
    """
    target_times = check_and_convert(target_times).astype(float).tolist()

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [maxtime, S(t|x)]
    # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    predict_probabilities = np.array(spline(target_times))
    for i, target_time in enumerate(target_times):
        if target_time > max_time:
            predict_probabilities[i] = max(slope * target_time + 1, 0)

    return predict_probabilities


def predict_mean_survival_time(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray
):
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the integral will be infinite.")
        return np.inf

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = max(times_coordinate.tolist())

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)],
    #                             max_time + (0 - np.array(spline(max_time)).item()) / slope)
    if 0 in survival_curve:
        zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)])
    else:
        zero_probability_time = max_time + (0 - np.array(spline(max_time)).item()) / slope

    def _func_to_integral(time, maximum_time, slope_rate):
        return np.array(spline(time)).item() if time < maximum_time else (1 + time * slope_rate)
    # _func_to_integral = lambda time: spline(time) if time < max_time else (1 + time * slope)
    # limit controls the subdivision intervals used in the adaptive algorithm.
    # Set it to 1000 is consistent with Haider's R code
    mean_survival_time, *rest = integrate.quad(_func_to_integral, 0, zero_probability_time,
                                               args=(max_time, slope), limit=1000)
    return mean_survival_time


def predict_median_survival_time(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray
):
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the median survival time will be infinite.")
        return np.inf

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    min_prob = min(spline(times_coordinate.tolist()))

    if 0.5 in survival_curve:
        median_probability_time = times_coordinate[np.where(survival_curve == 0.5)[0][0]]
    elif min_prob < 0.5:
        min_time_before_median = times_coordinate[np.where(survival_curve > 0.5)[0][-1]]
        max_time_after_median = times_coordinate[np.where(survival_curve < 0.5)[0][0]]

        prob_range = robjects.FloatVector(
            spline(np.linspace(min_time_before_median, max_time_after_median, num=1000).tolist()))
        time_range = robjects.FloatVector(np.linspace(min_time_before_median, max_time_after_median, num=1000))
        inverse_spline = r_splinefun(prob_range, time_range, method='hyman')
        # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
        median_probability_time = np.array(inverse_spline(0.5)).item()
    else:
        max_time = max(times_coordinate.tolist())
        slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)
        median_probability_time = max_time + (0.5 - np.array(spline(max_time)).item()) / slope

    return median_probability_time


def stratified_folds_survival(
        dataset: pd.DataFrame,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        number_folds: int = 5
):
    event_times, event_indicators = event_times.tolist(), event_indicators.tolist()
    assert len(event_indicators) == len(event_times)

    indicators_and_times = list(zip(event_indicators, event_times))
    sorted_idx = [i[0] for i in sorted(enumerate(indicators_and_times), key=lambda v: (v[1][0], v[1][1]))]

    folds = [[sorted_idx[0]], [sorted_idx[1]], [sorted_idx[2]], [sorted_idx[3]], [sorted_idx[4]]]
    for i in range(5, len(sorted_idx)):
        fold_number = i % number_folds
        folds[fold_number].append(sorted_idx[i])

    training_sets = [dataset.drop(folds[i], axis=0) for i in range(number_folds)]
    testing_sets = [dataset.iloc[folds[i], :] for i in range(number_folds)]

    cross_validation_set = list(zip(training_sets, testing_sets))
    return cross_validation_set


@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)
    km_linear_zero: float = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        if self.survival_probabilities[-1] != 0:
            slope = (area_probabilities[-1] - 1) / area_times[-1]
            zero_survival = -1 / slope
            area_times = np.append(area_times, zero_survival)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)
        self.km_linear_zero = -1 / ((1 - min(self.survival_probabilities))/(0 - max(self.survival_times)))

    def best_guess(self, censor_times: np.array):
        surv_prob = self.predict(censor_times)
        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )
        censor_area = (self.area_times[censor_indexes] - censor_times) * self.area_probabilities[censor_indexes - 1]
        censor_area += self.area[censor_indexes]
        return censor_times + censor_area / surv_prob

    def _km_linear_predict(self, times):
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))

        predict_prob = np.empty_like(times)
        before_last_time_idx = times <= max(self.survival_times)
        after_last_time_idx = times > max(self.survival_times)
        predict_prob[before_last_time_idx] = self.predict(times[before_last_time_idx])
        predict_prob[after_last_time_idx] = np.clip(1 + times[after_last_time_idx] * slope, a_min=0, a_max=None)
        # if time <= max(self.survival_times):
        #     predict_prob = self.predict(time)
        # else:
        #     predict_prob = max(1 + time * slope, 0)
        return predict_prob

    def _compute_best_guess(self, time: float):
        """
        Given a censor time, compute the decensor event time based on the residual mean survival time on KM curves.
        :param time:
        :return:
        """
        # Using integrate.quad from Scipy should be more accurate, but also making the program unbearably slow.
        # The compromised method uses numpy.trapz to approximate the integral using composite trapezoidal rule.
        time_range = np.linspace(time, self.km_linear_zero, 2000)
        if self.predict(time) == 0:
            best_guess = time
        else:
            best_guess = time + np.trapz(self._km_linear_predict(time_range), time_range) / self.predict(time)

        # best_guess = time + integrate.quad(self._km_linear_predict, time, self.km_linear_zero,
        #                                    limit=2000)[0] / self.predict(time)
        return best_guess

    def best_guess_revise(self, censor_times: np.array):
        bg_times = np.zeros_like(censor_times)
        for i in range(len(censor_times)):
            bg_times[i] = self._compute_best_guess(censor_times[i])
        return bg_times
