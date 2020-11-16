"""
Note:
    Please change class_text_to_int function to reflect the labels
    used in label_map.pbtxt file.

"""
import os
import io
import pdb
import argparse
import pandas as pd
import tensorflow as tf
import sys
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def class_text_to_int(row_label):
    if row_label == "hand":  # 'keyboard'
        return 1
    # comment upper if statement and uncomment these statements for multiple labelling
    # if row_label == FLAGS.label0:
    #   return 1                    # as in pbtxt file
    # elif row_label == FLAGS.label1:
    #   return 2                     # as in pbtxt file
    else:
        None


def get_image_path(data_dir, img_name):
    """
    Find path of image. Thanks to Carlos for unique naming convention.
    This logic is only possible because of that.
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_dir):
        for file in f:
            if img_name in file:
                files.append(os.path.join(r, file))

    if len(files) > 1:
        print("More than one image with same name is found")
        print(files)
        sys.exit()

    return os.path.dirname(files[0])


def create_tf_example(group, data_dir):
    path = get_image_path(data_dir, group.filename)
    with tf.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"png"
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        try:
            classes_text.append(row["class"].encode("utf8"))
        except:
            pdb.set_trace()

        classes.append(class_text_to_int(row["class"]))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


def parse_arguments():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser(description="Creates tfrecord from csv files")
    parser.add_argument(
        "--csv", metavar="c", nargs="+", required=True, help="full paths to csv files"
    )
    parser.add_argument(
        "--labels", metavar="l", nargs="+", required=True, help="Lable of objects"
    )
    parser.add_argument(
        "--output",
        metavar="o",
        type=str,
        required=True,
        help="full path to output file",
    )
    parser.add_argument(
        "--images",
        metavar="ip",
        nargs="+",
        required=False,
        help="Root directory whre the images can be found",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    writer = tf.python_io.TFRecordWriter(args.output)

    # Looping through all th csv files
    for idx, ccsv in enumerate(args.csv):
        print("Creating tfrecord for ", ccsv)
        data_dir = args.images[idx]
        ccsv_df = pd.read_csv(ccsv)
        grouped = split(ccsv_df, "filename")
        for group in grouped:
            tf_example = create_tf_example(group, data_dir)
            writer.write(tf_example.SerializeToString())

    writer.close()
    print("Successfully created tfrecord file at ", args.output)
