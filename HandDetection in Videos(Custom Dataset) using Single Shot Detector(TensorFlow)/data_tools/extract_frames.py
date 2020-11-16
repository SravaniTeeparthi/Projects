'''
DESCRIPTION:
    The following script extracts frames and corresponding bounding boxes
    from typing/notyping acitvity ground truth(`../docs/preparing-dataset.md)
    present under a root directory.

    Extracted frames and bonding boxes(csv file) are stored at output
    directory location. The bounding boxes csv file has following
    columns
        1. filename     : name of png image of current bounding box 
        2. width        : Width of the image
        3. height       : Height of the image
        4. class        : object class name(`keyboard`).
        5. (xmin, ymin) : Top left coordinates of bounding box.
        6. (xmax, ymax) : Top left coordinates of bounding box.

NOTE:
    It is assumed that ground truth csv files and videos are located in same
    directory
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import vlearn
from vlearn.torch_objdet import AOLMEData
import argparse
import pdb


def parse_arguments():
    cur_description = "DESCRIPTION:\n  The following script extracts frames and " +\
                      "corresponding bounding boxes from\n"+\
                      "  activity cubes ground truth "+\
                      "(Refer `../docs/preparing-dataset.md#AOLME`.)"
    # Parse arguments
    parser=argparse.ArgumentParser(
        description=cur_description,
    formatter_class=vlearn.ArgFormatter)

    parser.add_argument("root_dir", type=str,
                        help="root directory having ground truth csv files")
    parser.add_argument("output_dir", type=str,
                        help="output directory path")
    parser.add_argument("gt_csv_name", type=str,
                        help="Name of csv file")
    parser.add_argument("--frms_per_min", type=int, metavar="F", default=60,
                        help="Number of frames to extract every minute")
    parser.add_argument("--objclass", type=str, metavar="F", default="keyboard",
                        help="Object name, {keyboard, paper, hand}")

    args_dict             = {}
    args                  = parser.parse_args()
    args_dict["rdir"]     = args.root_dir
    args_dict["odir"]     = args.output_dir
    args_dict["fname"]    = args.gt_csv_name
    args_dict["frms_per_min"]    = args.frms_per_min
    args_dict["objclass"] = args.objclass
    
    return args_dict





# Initialize keyboard instance for AOLME Data
args_dict = parse_arguments()
kb_gt     = AOLMEData(**args_dict)

# Extracting frames and corresponding bounding boxes
kb_gt.extract_objdet_data()
