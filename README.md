# collision-detector
Using an RGB-D (depth) camera, detect people and measure the distance to them.  Originally created as a safety device for construction equipment.

![screenshot](/docs/screenshot.png?raw=true)


## Hardware Requirements

* Intel RealSense depth camera

## Software Requirements

* librealsense (https://github.com/IntelRealSense/librealsense.git)
* OpenCV 3.4 with DNN module


## To install requirements on FC-31:

```
sudo dnf install opencv-devel
sudo dnf install librealsense-devel
```
