"""
This script splits `keyboard_det_gt.csv`, created using `extract_frames.py` into
1. `trn_keyboard_det_gt.csv`
2. `val_keyboard_det_gt.csv`
3. `tst_keyboard_det_gt.csv`
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import vlearn
import pandas as pd
from fnmatch import fnmatch
import pdb
import os

# Argument parser
def parse_arguments():
    cur_description = "DESCRIPTION:Creates training, testing and validation splits"
    # Parse arguments
    parser=argparse.ArgumentParser(
        description=cur_description,
    formatter_class=vlearn.ArgFormatter)

    parser.add_argument("objdet_gt", type=str,
                        help="CSV containing bounding boxes")
    parser.add_argument("labels_csv", type=str,
                        help="CSV containing label(trn,tst or val) for each AOLME group")

    args_dict                = {}
    args                     = parser.parse_args()
    args_dict["labels_csv"]  = args.labels_csv
    args_dict["objdet_gt"]  = args.objdet_gt

    return args_dict


args_dict = parse_arguments()

objdet_df = pd.read_csv(args_dict["objdet_gt"])
labels_df = pd.read_csv(args_dict["labels_csv"])

for label in labels_df.label.unique():

    clabel_df = labels_df.copy()
    clabel_df = clabel_df[clabel_df["label"] == label]


    csplit_df = pd.DataFrame(columns=objdet_df.columns)
    for ridx, row in clabel_df.iterrows():

        cls   = row["group"].split("-")[0] # Cohort, Level, School
        grp   = row["group"].split("-")[1]
        date  = row["date"]
        #import pdb; pdb.set_trace()
        regex = "*" + cls + "-"+ str(date) +"-"+ grp + "-*"

        cgrp_df    = objdet_df.copy()
        valid_rows = [fnmatch(x,regex) for x in cgrp_df["filename"]]
        cgrp_df    = cgrp_df[valid_rows]

        csplit_df  = pd.concat([csplit_df, cgrp_df],ignore_index=True)

    csplit_csv_name = os.path.dirname(args_dict["objdet_gt"]) + "/" + label + ".csv"
    print("Writing, " + csplit_csv_name)
    csplit_df.to_csv(csplit_csv_name)
