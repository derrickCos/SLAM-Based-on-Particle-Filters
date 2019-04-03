# ECE 276A Project #2 - SLAM

Pengluo Wang,

University of California, San Diego, 2019

## Content

- [Overview](# overview)
- [Requirements & Installation](# requirements-&-installation)
- [Demo](# demo)
- [Results](# results)

##Overview

Implement simultaneous localization and mapping (SLAM) using odometry, inertial, 2-D laser range, and RGBD measurements from a differential-drive robot. IMU, odometry, and laser measurements have been used to localize the robot and build a 2-D occupancy grid map of the environment. RGBD information is used to map the texture onto the floor of the 2-D map.

IMU, odometry, and laser measurements are included in the project under `data` directory. Kinect RGBD information can be found at https://drive.google.com/open?id=1IR9KDzxeXFEdsEmVezcOx8jtcUx8PYax. Unzip and put the data directory `dataRGBD` under dir `data` in the project. Three datasets (20, 21, 23) are included in the project. However only the RGBD information of dataset 20 and 21 is available.

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

##Demo

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

## Results

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

