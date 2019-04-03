# SLAM Based on Particle Filters

Pengluo Wang,

University of California, San Diego, 2019

## Content

- [Overview](#overview)
- [Requirements & Installation](#requirements--installation)
- [Demo](#demo)
- [Results & Analysis](#results—analysis)

## Overview

Implement simultaneous localization and mapping (SLAM) using odometry, inertial, 2-D laser range, and RGBD measurements from a differential-drive robot. IMU, odometry, and laser measurements have been used to localize the robot and build a 2-D occupancy grid map of the environment. RGBD information is used to map the texture onto the floor of the 2-D map.

IMU, odometry, and laser measurements are included in the project under `data` directory. Kinect RGBD information can be found at https://drive.google.com/open?id=1IR9KDzxeXFEdsEmVezcOx8jtcUx8PYax. Unzip and put the data directory `dataRGBD` under dir `data` in the project. Three datasets (20, 21, 23) are included in the project. However only the RGBD information of dataset 20 and 21 is available.

The configuration of the robot has been given in [RobotConfiguration.pdf](/documents/RobotConfiguration.pdf). Notice that the pose matrix for each sensor is calculated directly from the robot configuration document without calibration. We assume each pose matrix is static within robot frame.

## Requirements & Installation

### Environment

- Python 3.7

### Requirements

- numpy
- scipy
- imageio
- matplotlib
- tqdm
- imageio-ffmpeg

### Installation

If you are using `Conda` for Python environment management:

```
conda create -n slam_env python==3.7
conda activate slam_env
pip install -U pip
pip install -r requirements.txt
```

## Demo

Run

```
python main.py -d 20
```

If using RGBD information to map texture onto the floor:

```
python main.py -d 20 -t
```

Some other optional arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  -d DATASET            dataset number, default 20
  -t                    plot texture information, default False
  -N N_PARTICLES        number of particles, default 100
  -n                    introduce NO noise for motion predict
  --sigma SIGMA         noisy level, default 0.5
  -r RESOLUTION         map resolution, default 0.1
  -f_i FRAME_INTERVAL   frame interval to save plots, default 15
  -f_th FLOOR_THRESHOLD
                        floor height threshold, default 0.15
```

## Results & Analysis

Detailed problem formulation and technical approach are included in [SLAM Report.pdf](/documents/SLAM Report.pdf). Results and analysis are shown below as well as in the report.

#### Dataset 20

<img src="results/20.gif" style="height:50%;width:50%;" />

The purple pixels are unknown cells (undetected by lisar), yellow pixels are occupied cells (wall), cyan pixels are free cells (corridors and rooms). Red triangle represents the robot and its orientation is pointed by one of the angle. Blue dots represent all the particles. Easy to see that the particles have reasonable locations (at the body of the robot). Also notice that loop closure is achieved.

#### Dataset 21:

<img src="results/21.gif" style="height:50%;width:50%;" />

#### Dataset 23:

<img src="results/23.gif" style="height:50%;width:50%;" />

Performance of dataset 23 is not so well at the ending of the trajectory because we assume the floor is flat and there is no shift in the z-axis (vertical dimension). However this assumption does not hold in practice for dataset 23. At the end of trajectory the robot is climbing up a slop, thus leading to bad performance.

#### Dataset 20 with texture:

<img src="results/20_texture.gif" style="height:80%;width:80%;" />

Result of floor texture mapping is reasonable from visual inspection.

#### Dataset 20 without noise:

<img src="results/20_no_noise.gif" style="height:50%;width:50%;" />

This result is obtained without introducing any noise during motion prediction step. As we can see the performance is much worse and loop closure isn't achieved in this case. This emphasizes the importance of introducing adequate amount of noises to reflect an actual motion prediction model in the real world. Another observation is that the performance gets worse when the robot is rotating, indicating that the error may be mainly introduced by IMU.