"""
Creates NodeInfoDict and AdjacencyDict objects for Reactome pathways.

Per Liang et al, only genes that participate in an edge are included in the NodeInfoDict. If the graph is to have
directed edges only, the node set will only include genes that participate in directed edges. If the graph is to
have both directed and undirected edges, then the node set will only exclude genes that do not participate in any edge.

The NodeInfoDict for the Reactome pathway graphs is populated from gene data scraped from the KEGG database when
possible, which is why this script relies on a NodeInfoDict scraped from KEGG. If a gene cannot be cross-referenced with
a KEGG entry, the values of the NodeInfoDict for the reactome pathway graphs are populated with basic information that
can be inferred without the KEGG database.

The Reactome graph data saved to disk is a tuple of four graphs. The first two items are empty graphs and are present
because the KEGG BRITE orthology contains four levels of hierarchically related graphs, and including empty
placeholders for the first items in the tuple ensures that downstream code can be recycled.
"""
import argparse
import os
import pandas as pd
import pickle
import tempfile

from tqdm import tqdm
from typing import Tuple

from custom_data_types import AdjacencyDict, NodeInfoDict


def format_pathway_dataframe(x: pd.DataFrame, directed: bool):
    
    if directed:
        x = x[x.direction == "directed"]
    else:
        # Convert all undirected edges to pairs of directed edges
        reversed_edges = x[x.direction == "undirected"]
        if len(reversed_edges) > 0:
            tmp_src, tmp_dest = reversed_edges.src.copy(), reversed_edges.dest.copy()
            # Swap the source and destination nodes
            reversed_edges.loc[:, ["src"]] = tmp_dest
            reversed_edges.loc[:, ["dest"]] = tmp_src
            # Append the new edges to the original x and change all edges to directed
            x = pd.concat([x, reversed_edges], axis=0)
            x.direction = "directed"

    x.sort_values(by=["src", "dest"], inplace=True)
    x = x.drop_duplicates(ignore_index=True).astype(str)
    x.src, x.dest = "hsa" + x.src, "hsa" + x.dest
    # Rename edge types that have illegal characters
    edge_type_map = {
        "Binding": "Binding",
        "Control(In: ACTIVATION of BiochemicalReaction)": "Control_In_ACTIVATION_of_BiochemicalReaction",
        "Control(In: ACTIVATION of Degradation)": "Control_In_ACTIVATION_of_Degradation",
        "Control(In: INHIBITION of BiochemicalReaction)": "Control_In_IHIBITION_of_BiochemicalReaction",
        "Control(Out: ACTIVATION of BiochemicalReaction)": "Control_Out_ACTIVATION_of_BiochemicalReaction",
        "Control(Out: ACTIVATION of TemplateReaction)": "Control_Out_ACTIVATION_of_TemplateReaction",
        "Control(Out: INHIBITION of BiochemicalReaction)": "Control_Out_INHIBITION_of_BiochemicalReaction",
        "Control(indirect)": "Control_indirect",
        "Process(BiochemicalReaction)": "Process_BiochemicalReaction"
    }
    renamed_edge_types = [edge_type_map[old_name] for old_name in x.type]
    x.type = renamed_edge_types
    
    return x


def create_reactome_graph_info(
    x: pd.DataFrame,
    feature_map: pd.DataFrame,
    node_info: NodeInfoDict
) -> Tuple[NodeInfoDict, AdjacencyDict]:
    """
    :param x: A DataFrame object containing edge information for a graph. The DataFrame must contain columns named src,
    dest, direction, and type. The src and dest columns list genes in the pathway by their ENTREZ ids. The src and
    dest columns respectively contain the ENTREZ IDs of source and target nodes for each edge (arbitrary for
    undirected edges). The direction column states whether the edge in each row is "directed" or "undirected". The
    type column lists edge types.
    :param feature_map: A DataFrame that lists equivalent identifiers for each gene.
    :param node_info: A dictionary of node information previously scraped from the KEGG database. For compatibility
    with downstream modeling scripts.
    :return:
    """
    # Note: this function only works for directed edges in the graphs. Undirected need to be represented as a pair of
    # opposing directed edges, and this function does not create opposing edges from dest to src.
    src_list = x.src.drop_duplicates().to_list()
    gene_symbols = dict(zip(feature_map.kegg, feature_map.symbol))
    reactome_node_info, adj = {}, {src: set() for src in src_list}

    for src in tqdm(src_list, total=len(src_list), desc="Creating gene-level node info and adjacency dictionaries"):
        # If node information already exists in node_info, use it. Otherwise, make a new entry with limited information
        if src not in reactome_node_info.keys():
            if src in node_info.keys():
                reactome_node_info[src] = node_info[src]
            else:
                try:
                    # If ID maps to a gene symbol, use it in the node info gene symbol field
                    symbol = gene_symbols[src]
                except KeyError:
                    symbol = None
                reactome_node_info[src] = {
                    "level": None,
                    "id": src.strip("hsa"),
                    "gene_symbol": symbol,
                    "name": None,
                    "kegg_orthology": None,
                    "orthology_name": None,
                    "accession": None,
                    "children": set()
                }
        # Get the subset of outgoing edges from src
        x_subset = x[x.src == src].loc[:, ["dest", "type"]]
        for dest in x_subset.dest:
            # Add destination node to reactome_node_info if it is not already present
            if dest not in reactome_node_info.keys():
                if dest in node_info.keys():
                    reactome_node_info[dest] = node_info[dest]
                else:
                    try:
                        symbol = gene_symbols[dest]
                    except KeyError:
                        symbol = None
                    reactome_node_info[dest] = {
                        "level": None,
                        "id": dest.strip("hsa"),
                        "gene_symbol": symbol,
                        "name": None,
                        "kegg_orthology": None,
                        "orthology_name": None,
                        "accession": None,
                        "children": set()
                    }
            # Add src -> dest edge with type information to adj
            new_edge = (dest, tuple(x_subset.type[x_subset.dest == dest].to_list()))
            adj[src].add(new_edge)

    return reactome_node_info, adj


def main():

    # region Parse args
    parser = argparse.ArgumentParser(
        description="Creates Reactome node information and adjacency dictionaries."
    )
    parser.add_argument(
        "--data_dir",
        help="The path to the data directory",
        default="data"
    )
    parser.add_argument(
        "--pathway_dir",
        help="The path to the top-level directory containing Reactome pathway data. Must contain a subdirectory "
             "called \"pathways\" that holds text files listing edges for each individual Reactome pathway.",
        default="data/Pathway"
    )
    parser.add_argument(
        "--directed",
        help="If set, only create a graph from directed edges. Otherwise, include both directed and undirected edges.",
        dest="directed", action="store_true"
    )
    parser.add_argument(
        "--merge_pathways",
        help="If set, this script will create NodeInfoDict and AdjacencyDict objects for a graph that contains all "
             "Reactome pathways as subgraphs. If this flag is not set, the default behavior is to create separate "
             "NodeInfoDict and AdjacencyDict objects for each Reactome pathway.",
        action="store_true"
    )
    # endregion Parse args

    args = vars(parser.parse_args())

    pathway_dir = args["pathway_dir"]
    pathway_subdir = os.path.join(pathway_dir, "pathways")
    pathway_files = os.listdir(pathway_subdir)

    brite_graph_file = os.path.join(args["data_dir"], "brite_graph.pkl")
    feature_map_file = os.path.join(args["data_dir"], "feature_map.csv")

    merge_pathways = True if args["merge_pathways"] else False
    merge = "merged" if merge_pathways else "unmerged"

    directed = True if args["directed"] else False
    direction = "directed" if directed else "undirected"

    feat_map = pd.read_csv(feature_map_file)

    # Get the node information dictionary that was created from BRITE
    with open(brite_graph_file, "rb") as file_in:
        brite_graph = pickle.load(file_in)
    brite_d_dict = brite_graph[3][0]

    # Create the pathway dictionary -- Selected pathways used by Liang et al. This is only used when training
    # models that feeds all pathway graphs into the model at the same time as a single large graph.
    level_c_dict = {}
    pathways_used = []  # Keep track of pathways used by Liang et al.
    file_path = os.path.join(pathway_dir, "pathway_genes_list.txt")
    lines_in = (line.split() for line in open(file_path, "r"))
    for line in lines_in:
        pathway_name = line[0]
        pathways_used.append(pathway_name)
        pathway_members = set("hsa" + entrez_id for entrez_id in line[1:])
        level_c_dict[pathway_name] = {
            "level": "C",
            "id": None,
            "gene_symbol": None,
            "name": None,
            "kegg_orthology": None,
            "orthology_name": None,
            "accession": None,
            "children": pathway_members
        }

    # Create gene-level NodeInfoDict and AdjacencyDict for a graph merged from all pathways. This information will
    # be used to create MLP input tensors regardless of whether the GNN survival models use merged graphs as input.

    # Write the edge data for all pathways into a single temporary file and then read it into memory as a DataFrame
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        tmp.write("src_type\tsrc\tdest_type\tdest\tdirection\ttype\n")
        for file in pathway_files:
            if file.strip(".txt") not in pathways_used:  # Only use pathways selected by Liang et al.
                continue
            abs_file_path = os.path.abspath(os.path.join(pathway_subdir, file))
            lines_in = (line for line in open(abs_file_path, "r"))
            next(iter(lines_in))  # Skip the column headers
            for line in lines_in:
                tmp.write(line)
        df = format_pathway_dataframe(pd.read_table(tmp.name), directed)
    finally:
        tmp.close()
        os.unlink(tmp.name)
    # Create the NodeInfoDict and AdjacencyDict for the merged graph of all pathways
    node_dict, edge_dict = create_reactome_graph_info(df, feature_map=feat_map, node_info=brite_d_dict)

    # Write to disk
    level_a = (dict(), dict())
    level_b = (dict(), dict())
    level_c = (level_c_dict, dict())
    level_d = (node_dict, edge_dict)
    graph_info = (level_a, level_b, level_c, level_d)
    file_name = f"reactome_graph_{direction}.pkl"
    path_out = os.path.join(args["data_dir"], file_name)
    with open(path_out, "wb") as file_out:
        pickle.dump(graph_info, file_out)

    # If the GNN survival models are to take individual pathway graphs as input, one (NodeInfoDict, AdjacencyDict)
    # per pathway must be written to disk.
    if not merge_pathways:
        # Make a subdirectory that houses the NodeInfoDict objects for individual Reactome pathway graphs
        output_dir = os.path.join(pathway_subdir, "dicts")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Create NodeInfoDict and AdjacencyDict objects for each individual Reactome pathway graph
        for file in pathway_files:
            if file.strip(".txt") not in pathways_used:
                continue  # Skip subdirectories and pathways not used by Liang et al.
            abs_file_path = os.path.abspath(os.path.join(pathway_subdir, file))
            df = format_pathway_dataframe(pd.read_table(abs_file_path), directed)
            node_dict, edge_dict = create_reactome_graph_info(df, feature_map=feat_map, node_info=brite_d_dict)
            graph_info = (node_dict, edge_dict)
            file_name = f"{file.strip('.txt')}_{direction}.pkl"
            path_out = os.path.join(output_dir, file_name)
            # Write to disk
            with open(path_out, "wb") as file_out:
                pickle.dump(graph_info, file_out)


if __name__=="__main__":
    main()