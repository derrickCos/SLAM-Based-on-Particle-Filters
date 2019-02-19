"""
ECE 276A WI19 HW2
SLAM and texture mapping
Author: Pengluo Wang
Date: 02/16/2019
"""
import os
import tqdm
import shutil
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from utils import load_and_process_data, generate_video
from mapping import Map
from robot import Robot


### Load data and create world & robot
dataset = 20
save_dir = os.path.join('results', str(dataset))
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
data_sync = load_and_process_data(dataset=dataset)
world = Map(-15, -15, 15, 15)
initial_state = [(0, 0), 0]
N_particles = 200
robot = Robot(initial_state, N_particles, mode='uniform')

# for idx_t in tqdm.trange(len(data_sync['stamps'])):
for idx_t in tqdm.trange(2000, 2100):
    # extract sensor data
    lidar_coords = data_sync['lidar_coords'][idx_t]
    encoder_counts = data_sync['encoder_counts'][idx_t]
    imu_yaw = data_sync['imu_yaw'][idx_t]
    dt = data_sync['dt'][idx_t]

    ### MAPPING: update mapping based on current lidar scan
    world.update(lidar_coords, robot.state)
    world.show(data_sync['stamps'][idx_t], robot.trajectory, robot.state[1])
    world.show_particles(world.ax, robot.particles)

    ### PREDICTION: use encoder and yaw info to update robot trajectory
    robot.advance_by(encoder_counts, imu_yaw, dt, noisy=True)

    ### UPDATE: update particle positions and weights using lidar scan
    robot.update_particles(lidar_coords, world.map, world.res, world.xmin, world.ymax)

    # plt.pause(1e-20)    # comment for faster iteration without displaying each frame
    plt.savefig(os.path.join(save_dir, 'result%05d.png' % idx_t), dpi=150)


generate_video(save_dir)
plt.show()  # if commented, figure will be closed after main script ends
