import numpy as np
import tensorflow as tf
import cv2
import sys
import glob
import os
import pdb
import skvideo.io
from utils import label_map_util

# Read video
vid_path = "/home/sravani/Dropbox/typing-notyping/C1L1P-E/20170302/G-C1L1P-Mar02-E-Irma_q2_02-08_30fps.mp4"
out_path = "C1L1P-E/20170302/G-C1L1P-Mar02-E-Irma_q2_02-08_30fps.mp4"
cap = cv2.VideoCapture(vid_path)
fps = round(cap.get(cv2.CAP_PROP_FPS))

output_params = {
    "-vcodec": "libx264",
    "-pix_fmt": "yuv420p",
    "-r": str(fps),
}
writer = skvideo.io.FFmpegWriter(out_path, outputdict=output_params)

# Read the graph.
with tf.gfile.FastGFile(
    "/home/sravani/Softwares/tensorflow-handdetection/handtracking/training_aolme_new/trained-inference-graphs/frozen_inference_graph.pb",
    "rb",
) as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Read labels protbuffer file
category_index = label_map_util.create_category_index_from_labelmap(
    "/home/sravani/Softwares/tensorflow-handdetection/handtracking/hand_inference_graph/hand_label_map.pbtxt",
    use_display_name=True,
)

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name="")

    # Video loop
    while cap.isOpened():
        ret, img = cap.read()
        img_lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_l       = img_lab[...,0]
        clahe       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_clahe     = clahe.apply(img_l)
        img_clahe   = cv2.merge((l_clahe,
                                 img_lab[...,1],
                                 img_lab[...,2]))
        img_clahe   = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

        if ret == True:
            rows = img_clahe.shape[0]
            cols = img_clahe.shape[1]
            inp = cv2.resize(img_clahe, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run(
                [
                    sess.graph.get_tensor_by_name("num_detections:0"),
                    sess.graph.get_tensor_by_name("detection_scores:0"),
                    sess.graph.get_tensor_by_name("detection_boxes:0"),
                    sess.graph.get_tensor_by_name("detection_classes:0"),
                ],
                feed_dict={
                    "image_tensor:0": inp.reshape(1, inp.shape[0], inp.shape[1], 3)
                },
            )

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                className = category_index[classId]["name"]
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.5:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(
                        img,
                        (int(x), int(y)),
                        (int(right), int(bottom)),
                        (0, 0, 255),
                        thickness=2,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        img,
                        className +":" +str(int(score*100))+"%",
                        (int(x), int(y)),
                        font,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            # Write results
            out_frm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            writer.writeFrame(out_frm)
        # Break the loop if video cannot be read
        else:
            break
        # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # Close writer
    writer.close()
