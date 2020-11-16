# Installation
## Create anaconda environment
	conda create --name tfobj tensorflow=1.14 tensorflow 
    gpu=1.14 cudatoolkit=10.0 protobuf pillow lxml tk 
    conda activate tfobj 
    conda install -c conda-forge pycocotools
    
## Opencv
    pip install opencv-python opencv-contrib-python

## Clone
	cd ~/Software
    git clone https://github.com/tensorflow/models
    mv models tfmodels

## Cocoapi
	cd ~/Downloads/ 
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    make
    cp -r pycocotools ~/Software/tfmodels/research/
    
## Protubuf Installation
	# From tensorflow/models/research/
    protoc object_detection/protos/*.proto --python_out=.
    
## Add Libraries to PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:/home/sravani/Softwares/tensorflow-handdetection/tfmodels/research:/home/sravani/Softwares/tensorflow-handdetection/tfmodels/research/slim
    
## Testing Installation
	conda activate tfobj
    cd ~/Software/tfmodels/research/
    python object_detection/builders/model_builder_test.py


# DataPreProcessing
## Extract frames from activity ground truth
	cd /home/sravani/Dropbox/Marios_Shared/HAQ-AOLME/software/AOLME-HAR/keyboard-det-and-tracking/torch-keyboard-detection/data_tools
    python extract_frames --root_dir --output_dir gt_filename --frames_per_min --gtlabel
    
    
## Use https://www.makesense.ai/ to do ground truth



## Export annotation in required format yolo, voc or csv file. Here I used csv format for ouput.


## Image Augmentation(h-flip)
I used the following library for image augmentation. 

	git clone https://github.com/Paperspace/DataAugmentationForObjectDetection.git
    
The following script also gives changed bounding box cordinates as output

	cd C:\Softwares\ImageAugmentation\DataAugmentationForObjectDetection
    python flip_aolmeframes.py



## SPlit into training, validation and testing
The following script splits the csv file into training, validation and testing respectively

	cd /home/sravani/Dropbox/Marios_Shared/HAQ-AOLME/software/AOLME-HAR/keyboard-det-and-tracking/torch-keyboard-detection/data_tools
    python split.py csv_file split.csv
    
    
# Training and Testing
## Creating tf record
	cd generate_tfrecord  
    bash run_generate_tfrecord.sh
* Number of training samples = 7067 instances of hand
* Number of eval samples = 2030 isntances of hand

## Training
 Change num of steps accordingly, change paths in config file

	cd training
	time python model_main.py --model_dir=training\
	--num_train_steps=50000\
	--pipeline_config_path=/home/sravani/Softwares/tensorflow-handdetection/handtracking/model-checkpoint/ssdmobilenetv1/ssd_mobilenet_v1_coco.config 


## Exporting Model
Once training is done, the model can be exported using

	python export_inference_graph.py --input_type image_tensor\
    --pipeline_config_path /home/sravani/Softwares/tensorflow-handdetection/handtracking/modelcheckpoint/ssdmobilenetv1/ssd_mobilenet_v1_coco.config\
    --trained_checkpoint_prefix /home/sravani/Softwares/tensorflow-handdetection/handtracking/training_aolme_new/training/model.ckpt-50000\ 
    --output_directory /home/sravani/Softwares/tensorflow-handdetection/handtracking/training_aolme_new/trained-inference-graphs

## Evaluation
Run evaluation and output in eval directory
	
    python model_main.py --checkpoint_dir training --model_dir .\
    --pipeline_config_path /home/sravani/Softwares/tensorflow-handdetection/handtracking/model-checkpoint/ssdmobilenetv1/ssd_mobilenet_v1_coco.config --run_once 
    
It will create eval directory and places tensorflow event file inside. This can be opend used tensorboard

	tensorboard --logdir=training
    tensorboard --logdir ./eval --port 6006
    
It can be opened in browser using tensorboard [http://localhost:6006/](http://localhost:6006/)

## Testing on Video
Change video_path, output_name, .pb path and .pbtxt path

	cd ..
    cd testing
    python tf_detect_hands.py	

It produces .mp4 with detection for every frame
    
