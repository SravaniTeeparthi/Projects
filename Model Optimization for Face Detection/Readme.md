# Model Optimization for Face Detection
## The main goal of this project is to develop a face detection method that works for multiple scales and differerent aspect ratios. The basic approach is based on generating proposal boxes, processing them through PCA and SVM and optimizing the pipelined model.

### Steps
  1. Generate ground truth on images
  2. Use Image ground truth to generate ground truth for boxes. 
  
      ``` 
      python generating_groundtruth.py
      ```
  3. Optimized model performance estimation.
  
      ``` 
      python classifier.py
      ```
  4. Optimize the final model and obtain results from the final classifier
  
       ``` 
      python finalclassifier.py
      ```
[Results are reporetd in the paper ] (https://github.com/SravaniTeeparthi29/Projects/blob/main/Model%20Optimization%20for%20Face%20Detection/Final_paper.pdf)
