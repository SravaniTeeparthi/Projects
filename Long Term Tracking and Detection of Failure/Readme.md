# Long Term Tracking and Detection of Failure
## Opencv- Tracking Algorithms
  * As part of background the methods tested on custom dataset (videos) are Boosting, MIL, MOSSE, MedianFlow, 
    TLD, KCF, GOTRUN and CSRT for both speed and long term performance on a 23min video.
  * Performace is evaluated using intersection over union 
  
 ## Methodology
  Here we experimented with 
  - Normalized Cross correlation ` python norm_cross.py method_name `
  - Normalized Cross Correaltion with restricted area  ` python norm_cross_res.py method_name `
  -  Normalized Cross Correaltion with template update  ` python3 run_tracker.py <vid> <pos> <method> `
  -  Normalized Cross Correaltion with template update and moving window  ` python3 run_tracker.py <vid> <pos> <method>`
  
  ## Evaluation
  Perfomance is Evalated using IOU ratio `python intersection_over_union.py video_path gt_csv method_csv`
  
  ## Failure Handling 
  Random Forest is trained with 1313 class s (success), 161 class f (failure).
  ```
  cd Classifiers/ 
    python classify.py 
  ```
   
   [Results are reported in the paper] (https://github.com/SravaniTeeparthi29/Projects/blob/main/Long%20Term%20Tracking%20and%20Detection%20of%20Failure/final_paper_tracking.pdf) 
