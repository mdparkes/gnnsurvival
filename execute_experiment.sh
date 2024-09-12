#! /bin/bash

# This script runs through a survival modeling experiment. The following general steps
# are taken:
#
# - Creates input Tensor objects for GNN, MLP
# - Performs hyperparameter tuning using GNNs without SAGPool layers
# - Performs SAGPool ratio tuning in GNNs that use SAGPool layers
# - Performs hyperparameter tuning using MLPs without feature selection
# - Trains cross-validated GNN survival models without SAGPool
# - Trains cross-validated GNN survival models with SAGPool
# - Counts the frequency that each gene survived SAGPooling over all biopsies
# - Trains cross-validated MLP survival models without feature selection
# - Trains cross-validated MLP survival models with SAGPool-based feat. sel.
# - Exports the results of the experiment in CSV files
#
# This experiment deals with directed, non-relational Reactome graphs. The pathway graphs are
# unmerged; that is, one pathway graph is fed into the GNN at a time instead of a large
# single graph containing all the pathway graphs as subgraphs.

DATA_DIR=data
PATHWAY_DIR=${DATA_DIR}/Pathway
EXPRS_FILE=${DATA_DIR}/tcga_exprs.csv
CLIN_FILE=${DATA_DIR}/tcga_clin.csv
N_FOLDS=6
N_INTERVALS=2
BATCH_SIZE=48
N_WORKERS=4
# BATCH_SIZE / N_WORKERS = number of minibatch samples handled by each worker

while getopts "e:t:" arg; do
	case "${arg}" in
		e) EXPT_DIR=${OPTARG};;
		t) CANCER=${OPTARG};;
	esac
done


CANCER_U=$(echo "${CANCER}" | tr "[:lower:]" "[:upper:]")
CANCER_L=$(echo "${CANCER}" | tr "[:upper:]" "[:lower:]")


export PYTHONPATH="${PYTHONPATH}:/gnnsurvival-pyg"


[ -z  "$(ls -A ${DATA_DIR}/${CANCER_U}/graphs/reactome/unmerged/directed/nonrelational/raw)" ] && \
  echo "Creating PyG Data objects for Reactome pathways" && \
  python3 src/create_pyg_graph_objs.py \
	  --output_dir ${DATA_DIR} \
	  --cancer_types ${CANCER_U} \
	  --database reactome \
	  --nonrelational \
	  --directed \
	  --kfolds ${N_FOLDS}


[ -z "$(ls -A ${DATA_DIR}/${CANCER_U}/mlp_inputs/reactome/directed/raw)" ] && \
  echo "Creating MLP input data" && \
  python3 src/create_mlp_inputs.py \
 	  --output_dir ${DATA_DIR} \
 	  --cancer_types ${CANCER_U} \
 	  --database reactome \
 	  --directed \
 	  --kfolds ${N_FOLDS}


[ "$(ls -A experiment1/${CANCER_U}/hyperparameters/gnn/reactome/unmerged/directed/nonrelational/*.pkl)" ] && \
  mkdir -p ${EXPT_DIR}/${CANCER_U}/hyperparameters/gnn/reactome/unmerged/directed/nonrelational && \
  cp -r experiment1/${CANCER_U}/hyperparameters/gnn/reactome/unmerged/directed/nonrelational \
  ${EXPT_DIR}/${CANCER_U}/hyperparameters/gnn/reactome/unmerged/directed/

[ -z "$(ls -A ${EXPT_DIR}/${CANCER_U}/hyperparameters/gnn/reactome/unmerged/directed/nonrelational/*.pkl)" ] && \
  echo "Tuning GNN hyperparameters" && \
  echo "Tuning hyperparameters of GNN without SAGPool layers" && \
  python3 src/tune_gnn_hyperparameters.py \
	  --data_dir ${DATA_DIR} \
	  --output_dir ${EXPT_DIR} \
	  --cancer_type ${CANCER_U} \
 	  --database reactome \
 	  --nonrelational \
 	  --directed \
 	  --batch_size ${BATCH_SIZE} \
	  --n_intervals ${N_INTERVALS} && \
  echo "Tuning SAGPool ratio" && \
  python3 src/tune_gnn_hyperparameters.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${EXPT_DIR} \
    --cancer_type ${CANCER_U} \
    --database reactome \
    --nonrelational \
    --directed \
    --batch_size ${BATCH_SIZE} \
    --n_intervals ${N_INTERVALS} \
    --sagpool


[ "$(ls -A experiment1/${CANCER_U}/hyperparameters/mlp/reactome/directed/*.pkl)" ] && \
  mkdir -p ${EXPT_DIR}/${CANCER_U}/hyperparameters/mlp/reactome/directed && \
  cp -r experiment1/${CANCER_U}/hyperparameters/mlp/reactome/directed \
  ${EXPT_DIR}/${CANCER_U}/hyperparameters/mlp/reactome/

[ -z "$(ls -A ${EXPT_DIR}/${CANCER_U}/hyperparameters/mlp/reactome/directed/*.pkl)" ] && \
  echo "Tuning MLP hyperparameters" && \
  python3 src/tune_mlp_hyperparameters.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${EXPT_DIR} \
    --cancer_type ${CANCER_U} \
    --database reactome \
    --directed \
    --batch_size ${BATCH_SIZE} \
    --n_intervals ${N_INTERVALS}


echo "Training GNN survival models without SAGPool"
python3 src/train_gnn_models.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--nonrelational \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--n_intervals ${N_INTERVALS} \
	--num_workers ${N_WORKERS} \
	--use_clin_feats \
 --normalize_gene_exprs


echo "Training GNN survival models with SAGPool"
python3 src/train_gnn_models.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--nonrelational \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--n_intervals ${N_INTERVALS} \
	--num_workers ${N_WORKERS} \
	--sagpool \
	--use_clin_feats \
	--normalize_gene_exprs


echo "Counting SAGPool gene retention frequencies"
python3 src/count_sagpool_selections.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--nonrelational \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--use_clin_feats \
	--normalize_gene_exprs

echo "Training MLP survival models without feature selection"
python3 src/train_mlp_models.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--n_intervals ${N_INTERVALS} \
	--num_workers ${N_WORKERS} \
	--use_clin_feats \
	--normalize_gene_exprs

echo "Training MLP survival models with SAGPool-based feature selection"
python3 src/train_mlp_models.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--n_intervals ${N_INTERVALS} \
	--num_workers ${N_WORKERS} \
	--sagpool \
	--use_clin_feats \
	--normalize_gene_exprs

echo "Exporting experiment results"
python3 src/export_gnn_experiment_data.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--nonrelational \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--use_clin_feats \
	--normalize_gene_exprs

python3 src/export_gnn_experiment_data.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--nonrelational \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--sagpool \
	--use_clin_feats \
	--normalize_gene_exprs

python3 src/export_mlp_experiment_data.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--use_clin_feats \
	--normalize_gene_exprs

python3 src/export_mlp_experiment_data.py \
	--data_dir ${DATA_DIR} \
	--output_dir ${EXPT_DIR} \
	--cancer_type ${CANCER_U} \
	--database reactome \
	--directed \
	--batch_size ${BATCH_SIZE} \
	--sagpool \
	--use_clin_feats \
	--normalize_gene_exprs
