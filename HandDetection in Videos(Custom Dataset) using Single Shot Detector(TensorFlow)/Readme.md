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
    
