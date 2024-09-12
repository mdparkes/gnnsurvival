import argparse
import pandas as pd
import re

def main():
    parser = argparse.ArgumentParser(
        description="Creates csv files containing RNAseq and clinical data. Records in the files can be matched "
                    "one-to-one on patient barcodes."
    )
    parser.add_argument("-ei", "--exprs_file",
                        help="The path to the file containing the RNA expression data",
                        default="data/tcga_exprs.tsv",
                        type=str)
    parser.add_argument("-ci", "--clin_file",
                        help="The path to the directory containing the clinical data",
                        default="data/clinical_PANCAN_patient_with_followup.tsv",
                        type=str)
    parser.add_argument("-d", "--dataset",
                        help="The type of dataset. Accepts \"TCGA\" or \"SCANB\" as arguments.",
                        default=None,
                        type=str)

    args = vars(parser.parse_args())
    exprs_file_in = args["exprs_file"]
    clin_file = args["clin_file"]
    dataset = args["dataset"]

    exprs_file_out = re.sub(r"\.tsv", ".csv", exprs_file_in)

    print("Loading gene expression data", end="... ", flush=True)
    exprs_data = pd.read_csv(exprs_file_in, index_col=0, sep="\t")
    print("Done", flush=True)

    print("Loading clinical data", end="... ", flush=True)
    index_col = "aliquot_submitter_id" if dataset == "TCGA" else "GEX.assay"
    clin_data = pd.read_csv(clin_file, encoding="windows-1252", index_col=index_col, low_memory=False)
    print("Done", flush=True)

    print("Cleaning up data and writing to disk", end="... ", flush=True)
    exprs_data = exprs_data.transpose()
    clin_data = clin_data[clin_data.index.isin(exprs_data.index)]
    exprs_data = exprs_data[exprs_data.index.isin(clin_data.index)]  # Use expression data that have clinical data
    clin_data = clin_data.loc[exprs_data.index, :]  # Make sure the order of the rows match

    exprs_data.to_csv(exprs_file_out)
    clin_data.to_csv(clin_file, index_label=index_col)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
