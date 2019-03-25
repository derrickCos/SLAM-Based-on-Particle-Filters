# ECE 276A Project #2 - SLAM

Pengluo Wang,

University of California, San Diego, 2019

### Overview

Implement simultaneous localization and mapping (SLAM) using odometry, inertial, 2-D laser range, and RGBD measurements from a differential-drive robot. Use the IMU, odometry, and laser measurements to localize the robot and build a 2-D occupancy grid map of the environment. Use the RGBD information to color the floor of the 2-D map.

IMU, odometry, and laser measurements are included in the project. Kinect RGBD information can be found at https://drive.google.com/open?id=1IR9KDzxeXFEdsEmVezcOx8jtcUx8PYax. Unzip and put the data directory `dataRGBD` under dir `data` in the project. Three dataset are included in the project, dataset 20, 21, and 23. However only the RGBD information of dataset 20 and 21 is available.

### Requirements

- Python 3.7

### Installation

```
conda create -n slam_env python==3.7
conda activate slam_env
pip install -U pip
pip install -r requirements.txt
```

### Demo

Run

```
python main.py -d 20
```

If using RGBD information to map texture:

```
python main.py -d 20 -t
```

### Results

Dataset 20:

<img src="results/20.gif" style="height:50%;width:50%;" />

Dataset 21:

<img src="results/21.gif" style="height:50%;width:50%;" />

Dataset 23:

<img src="results/23.gif" style="height:50%;width:50%;" />

Dataset 20 with texture:

<img src="results/20_texture.gif" style="height:80%;width:80%;" />

Dataset 21 with texture:

<img src="results/21_texture.gif" style="height:80%;width:80%;" />

Dataset 20 without noise:

<img src="results/20_no_noise.gif" style="height:50%;width:50%;" />

