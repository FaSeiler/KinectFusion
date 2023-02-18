# KinectFusion

============

This is an implementation of KinectFusion, based on _Newcombe, Richard A., et al._
**KinectFusion: Real-time dense surface mapping and tracking.**

Dependencies
------------
* **CUDA 12.0**. In order to provide real-time reconstruction.
* **OpenCV 4.7**. It requires to build OpenCV 4.7 with CUDA.
* **Eigen3** for efficient matrix and vector operations.
* **Glog**.
* **Ceres**.
* **Flann** for nearest neighbor search.
* **FreeImage**. 

Prerequisites
-------------
* Tum Dataset: https://vision.in.tum.de/data/datasets/rgbd-dataset/download
* Visual Studio 22
* GPU capable system
* Set custom opencv path (if necessary):
SET("OpenCV_DIR" "C:/KinectFusion/Development/Libs/opencv-4.7.0/Build")

Usage
-----

Path to the tum dataset would need to be adapted 

```Cpp

    if (DATASET_FREIBURG1_XYZ) {
        filenameIn = std::string("../../Data/rgbd_dataset_freiburg1_xyz/");
    }
    else if (DATASET_FREIBURG2_XYZ) {
        filenameIn = std::string("../../Data/rgbd_dataset_freiburg2_xyz/");
    }
    else {
        filenameIn = std::string("../../Data/rgbd_dataset_freiburg1_xyz/");
    }
```


