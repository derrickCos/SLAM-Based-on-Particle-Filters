"""
ECE 276A WI19 HW2
SLAM and texture mapping
Author: Pengluo Wang
Date: 02/16/2019
"""
import os
import tqdm
import shutil
import numpy as np
import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from utils import load_and_process_data, generate_video
from mapping import Map
from robot import Robot


### Load data and create world & robot
dataset = 20
rgbd_dir = os.path.join('data', 'dataRGBD')
save_dir = os.path.join('results', str(dataset))
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
data_sync = load_and_process_data(dataset=dataset, texture='on')
world = Map(-10, -10, 15, 15, res=0.05)
initial_state = ((0, 0), 0)
N_particles = 200
robot = Robot(initial_state, N_particles)
# initialize map
world.update_map(data_sync['lidar_coords'][0], np.matmul(robot.T_wb, robot.T_bl))

for idx_t in tqdm.trange(1, len(data_sync['stamps']), desc='Progress', unit='frame'):
# for idx_t in tqdm.trange(1200, 1400):
    # extract sensor data
    lidar_coords = data_sync['lidar_coords'][idx_t]
    encoder_counts = data_sync['encoder_counts'][idx_t]
    imu_yaw = data_sync['imu_yaw'][idx_t]
    dt = data_sync['dt'][idx_t]

    ### PREDICTION: use encoder and yaw info to update robot trajectory
    robot.advance_by(encoder_counts, imu_yaw, dt, noisy=True)

    ### UPDATE: update particle positions and weights using lidar scan
    robot.update_particles(lidar_coords, world.grid_map, world.res, world.xmin, world.ymax)

    ### MAPPING: update mapping based on current lidar scan
    world.update_map(lidar_coords, np.matmul(robot.T_wb, robot.T_bl))

    if data_sync['rgb_update'][idx_t] and data_sync['disp_update'][idx_t]:
        rgb = plt.imread(os.path.join(rgbd_dir, data_sync['rgb_file_path'][idx_t]))
        disp = np.array(PIL.Image.open(os.path.join(rgbd_dir, data_sync['disp_file_path'][idx_t])))
        yaw = robot.state[1]
        R_wb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
        p_wb = np.array(robot.state[0] + (0.177,))
        T_wb =np.vstack((np.hstack((R_wb, p_wb.reshape(3, 1))),
                         np.array([[0, 0, 0, 1]])))
        world.update_texture(rgb, disp, robot.K_oi, np.matmul(T_wb, robot.T_bo))

    if idx_t % 15 == 0:
        # display every 15 frames
        world.show(data_sync['stamps'][idx_t], robot.trajectory, robot.state[1])
        world.show_particles(world.ax1, robot.particles)
        # plt.pause(1e-20)    # comment for faster iteration without displaying each frame
        plt.savefig(os.path.join(save_dir, 'result%05d.png' % idx_t), dpi=150)


generate_video(save_dir)
plt.show()  # if commented, figure will be closed after main script ends
