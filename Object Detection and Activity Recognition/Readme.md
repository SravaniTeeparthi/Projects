## The goal of this project is to identify paper and localize "writing,no-writing" in a given video using optimized 3D-CNN Architectures.
  The architectures studied as part of this project are
  * SingleFB: Video + Single FilterBank CNN 
  * PyramidFB: Video + Pyramid FB CNN
  * InvPyramidFB: Video + Inverted Pyramid FB CNN
  * OFSingleFB: Optical Flow + Single FB CNN 
  
  The winning architecture is picked from 21 random runs.
  * For object detection, Faster RCNN is re-trained on custom dataset. 
