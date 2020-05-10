# 3D-Object-TrackingSFND 3D Object Tracking
Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.



In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks:

First, you will develop a way to match 3D objects over time by using keypoint correspondences.
Second, you will compute the TTC based on Lidar measurements.
You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches.
And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course.
Dependencies for Running Locally
cmake >= 2.8
All OSes: click here for installation instructions
make >= 4.1 (Linux, Mac), 3.81 (Windows)
Linux: make is installed by default on most Linux distros
Mac: install Xcode command line tools to get make
Windows: Click here for installation instructions
Git LFS
Weight files are handled using LFS
OpenCV >= 4.1
This must be compiled from source using the -D OPENCV_ENABLE_NONFREE=ON cmake flag for testing the SIFT and SURF detectors.
The OpenCV 4.1.0 source code can be found here
gcc/g++ >= 5.4
Linux: gcc / g++ is installed by default on most Linux distros
Mac: same deal as make - install Xcode command line tools
Windows: recommend using MinGW
Basic Build Instructions
Clone this repo.
Make a build directory in the top level project directory: mkdir build && cd build
Compile: cmake .. && make
Run it: ./3D_object_tracking.
