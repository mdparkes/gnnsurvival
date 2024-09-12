#! /bin/bash

DATA_DIR=data
CLIN_FILE=${DATA_DIR}/tcga_clin.csv
EXPRS_FILE=${DATA_DIR}/tcga_exprs.csv
PATHWAY_DIR=${DATA_DIR}/Pathway


export PYTHONPATH="${PYTHONPATH}:/gnnsurvival-pyg"

[ ! -f ${DATA_DIR}/tcga_exprs.tsv ] && \
  echo "Downloading TCGA gene expression data" && \
  FILE_URL=https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611 && \
  curl ${FILE_URL} -o ${DATA_DIR}/tcga_exprs.tsv && \


[ ! -f ${DATA_DIR}/tcga_exprs.csv ] && \
  echo "Creating CSV file from TCGA gene expression data" && \
  python3 src/create_csv_datasets.py \
    --exprs_file ${DATA_DIR}/tcga_exprs.tsv \
    --clin_file ${DATA_DIR}/tcga_clin.csv \
    --dataset TCGA


[ ! -f ${DATA_DIR}/brite_graph.pkl ] && \
  echo "Scraping BRITE graph data" && \
  python3 -m keggpathwaygraphs.create_graph \
    --output_dir ${DATA_DIR}


[ ! -f ${DATA_DIR}/reactome_graph_directed.pkl ] && \
  echo "Creating graph information for Reactome pathways" && \
  python3 src/create_reactome_graph.py \
    --data_dir ${DATA_DIR} \
    --pathway_dir ${PATHWAY_DIR} \
    --directed
