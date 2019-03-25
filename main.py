"""
ECE 276A WI19 HW2
SLAM and texture mapping
Author: Pengluo Wang
Date: 02/16/2019
"""
import os
import tqdm
import argparse
import numpy as np
import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from utils import check_and_rename, load_and_process_data, generate_video, WALL
from mapping import Map
from robot import Robot

parser = argparse.ArgumentParser(description='FastSLAM -- ECE 276A Project #2')
parser.add_argument('-d', dest='dataset', type=int, default=20, help='dataset number')
parser.add_argument('-t', dest='texture_on', action='store_true', help='plot texture information')
parser.add_argument('-N', dest='N_particles', type=int, default=100, help='number of particles')
parser.add_argument('-n', dest='no_noise', action='store_true', help='introduce NO noise for motion predict')
parser.add_argument('--sigma', dest='sigma', type=float, default=0.5, help='noisy level')
parser.add_argument('-r', dest='resolution', type=float, default=0.1, help='map resolution')
parser.add_argument('-f_i', dest='frame_interval', type=int, default=15, help='frame interval to save plots')
parser.add_argument('-f_th', dest='floor_threshold', type=float, default=0.15, help='floor height threshold')



if __name__ == '__main__':
    args = parser.parse_args()
    print('Configuration:')
    print('    ', args)

    ### Load data and create world & robot
    rgbd_dir = os.path.join('data', 'dataRGBD')
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    plots_save_path = os.path.join(result_dir, str(args.dataset))
    plots_save_path = check_and_rename(plots_save_path)
    os.mkdir(plots_save_path)
    data = load_and_process_data(dataset=args.dataset, texture_on=args.texture_on)
    world = Map(-10, -10, 15, 15, res=args.resolution, texture_on=args.texture_on)
    initial_state = ((0, 0), 0)
    N_particles = args.N_particles
    robot = Robot(initial_state, N_particles)
    # initialize map
    world.update_map(data['lidar_coords'][0], np.matmul(robot.T_wb, robot.T_bl))

    for idx_t in tqdm.trange(1, len(data['stamps']), desc='Progress', unit='frame'):
    # for idx_t in tqdm.trange(1000, 1050):
        # extract sensor data
        lidar_coords = data['lidar_coords'][idx_t]
        encoder_v = data['encoder_v'][idx_t]
        imu_w = data['imu_w'][idx_t]
        dt = data['dt'][idx_t]

        ### PREDICTION: use encoder and yaw info to update robot trajectory
        robot.advance_by(encoder_v, imu_w, dt, noisy=not args.no_noise, nv=args.sigma, nw=args.sigma)

        ### UPDATE: update particle positions and weights using lidar scan
        robot.update_particles(lidar_coords, 2*(world.grid_map == WALL) - 1, world.res, world.xmin, world.ymax)

        ### MAPPING: update mapping based on current lidar scan
        world.update_map(lidar_coords, np.matmul(robot.T_wb, robot.T_bl))

        if args.texture_on and data['rgb_update'][idx_t] and data['disp_update'][idx_t]:
            rgb = plt.imread(os.path.join(rgbd_dir, data['rgb_file_path'][idx_t]))
            disp = np.array(PIL.Image.open(os.path.join(rgbd_dir, data['disp_file_path'][idx_t])))
            yaw = robot.state[1]
            R_wb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])
            p_wb = np.array(robot.state[0] + (0.177,))
            T_wb =np.vstack((np.hstack((R_wb, p_wb.reshape(3, 1))),
                             np.array([[0, 0, 0, 1]])))
            world.update_texture(rgb, disp, robot.K_oi, np.matmul(T_wb, robot.T_bo), args.floor_threshold)

        if idx_t % args.frame_interval == 0:
            # display every 15 frames
            world.show(data['stamps'][idx_t], robot.trajectory, robot.state[1])
            # world.show_particles(world.ax1, robot.particles)
            # plt.pause(1e-20)    # commented for faster iteration without displaying plot
            plt.savefig(os.path.join(plots_save_path, 'result%05d.png' % idx_t), dpi=150)

    result_save_path = os.path.join(result_dir, str(args.dataset))
    if args.no_noise:
        result_save_path += '_no_noise'
    if args.texture_on:
        result_save_path += '_texture'

    result_save_path = check_and_rename(result_save_path, format='.gif')
    generate_video(plots_save_path, result_save_path)

    plt.show()  # if commented, main script will exit without displaying the final result


