import os
import imageio
import numpy as np
from scipy.signal import butter, lfilter

# map label
UNKNOWN = 0
WALL    = 2
FREE    = 1

# noise level for different
sigma_speed = 0.1
sigma_yaw = 0.01


def load_and_process_data(dataset, texture='on'):
    ## Load dataset
    with np.load(os.path.join('data', 'Encoders%d.npz' % dataset)) as data:
        encoder_counts = data["counts"]         # 4 x n encoder counts
        encoder_stamps = data["time_stamps"]    # encoder time stamps

    with np.load(os.path.join('data', 'Hokuyo%d.npz' % dataset)) as data:
        lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
        lidar_range_min = data["range_min"]  # minimum range value [m]
        lidar_range_max = data["range_max"]  # maximum range value [m]
        lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

    with np.load(os.path.join('data', 'Imu%d.npz' % dataset)) as data:
        imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    if texture == 'on':
        with np.load(os.path.join('data', 'Kinect%d.npz' % dataset)) as data:
            disp_stamps = data["disparity_time_stamps"]     # acquisition times of the disparity images
            rgb_stamps = data["rgb_time_stamps"]            # acquisition times of the rgb images


    ## Synchronize data
    reference_stamps = lidar_stamps     # use lidar_stamps as reference time
    len_stamps = len(reference_stamps)
    data = {
        'stamps': reference_stamps,
        't0': reference_stamps[0],
        'dt': np.zeros(np.size(reference_stamps)),
        'encoder_update': [False] * len_stamps,
        'encoder_counts': np.zeros((len_stamps, 4)),
        'lidar_update': [False] * len_stamps,
        'lidar_coords': [[]] * len_stamps,
        'imu_update': [False] * len_stamps,
        'imu_yaw': np.zeros(len_stamps),
        'rgb_update': [False] * len_stamps,
        'rgb_file_path': [[]] * len_stamps,
        'disp_update': [False] * len_stamps,
        'disp_file_path': [[]] * len_stamps
    }
    data['stamps'] = data['stamps'] - data['t0']
    data['dt'][1:] = data['stamps'][1:] - data['stamps'][:-1]

    # sync encoder
    encoder_stamps = encoder_stamps - data['t0']
    encoder_stamps = encoder_stamps[encoder_stamps >= 0]
    idx_t = 0
    idx_l = 0
    while True:
        idx_r = idx_l
        while idx_r < len(encoder_stamps) and encoder_stamps[idx_r] <= data['stamps'][idx_t]:
            idx_r = idx_r + 1
        if idx_r != idx_l:
            data['encoder_update'][idx_t] = True
            data['encoder_counts'][idx_t] = np.mean(encoder_counts[:, idx_l:idx_r],axis=1)
            idx_l = idx_r
        idx_t = idx_t + 1

        if idx_r >= len(encoder_stamps) or idx_t >= len(data['stamps']):
            break

    # sync lidar
    lidar_stamps = lidar_stamps - data['t0']
    lidar_stamps = lidar_stamps[lidar_stamps >= 0]
    lidar_angle = np.arange(lidar_angle_min, lidar_angle_max + lidar_angle_increment/2, lidar_angle_increment)
    idx_t = 0
    idx_l = 0
    while True:
        idx_r = idx_l
        while idx_r < len(lidar_stamps) and lidar_stamps[idx_r] <= data['stamps'][idx_t]:
            idx_r = idx_r + 1
        if idx_r != idx_l:
            data['lidar_update'][idx_t] = True
            lidar_range = np.mean(lidar_ranges[:, idx_l:idx_r], axis=1)
            # remove points too close or too far
            idxValid = np.logical_and((lidar_range < lidar_range_max),
                                      (lidar_range > lidar_range_min))
            valid_range = lidar_range[idxValid]
            valid_angle = lidar_angle[idxValid]
            x_so = valid_range * np.cos(valid_angle)
            y_so = valid_range * np.sin(valid_angle)
            data['lidar_coords'][idx_t] = np.vstack((x_so, y_so))
            idx_l = idx_r
        idx_t = idx_t + 1

        if idx_r >= len(lidar_stamps) or idx_t >= len(data['stamps']):
            break

    # sync imu
    imu_stamps = imu_stamps - data['t0']
    imu_stamps = imu_stamps[imu_stamps >= 0]
    # lpf
    def low_pass_filtering(data):
        cutoff = 10  # stop band: 10 Hz
        fs = 1 / np.mean(imu_stamps[1:] - imu_stamps[:-1])  # sampling rate
        order = 10
        normal_cutoff = 2 * cutoff / fs
        b, a = butter(order, normal_cutoff)
        return lfilter(b, a, imu_yaw)

    imu_yaw = imu_angular_velocity[2]
    imu_yaw_lp = low_pass_filtering(imu_yaw)
    idx_t = 0
    idx_l = 0
    while True:
        idx_r = idx_l
        while idx_r < len(imu_stamps) and imu_stamps[idx_r] <= data['stamps'][idx_t]:
            idx_r = idx_r + 1
        if idx_r != idx_l:
            data['imu_update'][idx_t] = True
            data['imu_yaw'][idx_t] = np.mean(imu_yaw_lp[idx_l:idx_r])
            idx_l = idx_r
        idx_t = idx_t + 1

        if idx_r >= len(imu_stamps) or idx_t >= len(data['stamps']):
            break

    # sync rgb
    if texture == 'on':
        rgb_stamps = rgb_stamps - data['t0']
        rgb_stamps = rgb_stamps[rgb_stamps >= 0]
        idx_t = 0
        idx_l = 0
        cnt = 0
        while True:
            idx_r = idx_l
            while idx_r < len(rgb_stamps) and rgb_stamps[idx_r] <= data['stamps'][idx_t]:
                idx_r = idx_r + 1
            if idx_r != idx_l:
                data['rgb_update'][idx_t] = True
                cnt = cnt + 1
                data['rgb_file_path'][idx_t] = os.path.join('RGB%d' % dataset, 'rgb%d_%d.png' % (dataset, cnt))
                idx_l = idx_r
            idx_t = idx_t + 1

            if idx_r >= len(rgb_stamps) or idx_t >= len(data['stamps']):
                break

    # sync disp
    if texture == 'on':
        disp_stamps = disp_stamps - data['t0']
        disp_stamps = disp_stamps[disp_stamps >= 0]
        idx_t = 0
        idx_l = 0
        cnt = 0
        while True:
            idx_r = idx_l
            while idx_r < len(disp_stamps) and disp_stamps[idx_r] <= data['stamps'][idx_t]:
                idx_r = idx_r + 1
            if idx_r != idx_l:
                data['disp_update'][idx_t] = True
                cnt = cnt + 1
                data['disp_file_path'][idx_t] = os.path.join('Disparity%d' % dataset, 'disparity%d_%d.png' % (dataset, cnt))
                idx_l = idx_r
            idx_t = idx_t + 1

            if idx_r >= len(disp_stamps) or idx_t >= len(data['stamps']):
                break

    return data


def add_noise(coords, sigma):
    return coords + sigma*np.random.uniform(-1, 1, *np.shape(coords))


def to_homo(coords):
    # coords: size should be (x, *)
    return np.vstack((coords, np.ones((1, coords.shape[1]))))


def to_non_homo(coords):
    # coords: size should be (x + 1, *)
    return coords[:-1] / coords[-1]


def rc_to_uv(coords, rmax):
    # coord: size should be (2, *)
    # return np.vstack((coords[1], rmax - 1 - coords[0])).squeeze().astype(int)
    return np.vstack((coords[1], coords[0])).squeeze().astype(int)


def world_to_image(coords, xmin, ymax, res):
    # coord: size should be (2, *)
    # xmin, ymax, res: properties of grid_map
    col = np.ceil((coords[0] - xmin)/res)
    row = np.ceil((ymax- coords[1])/res)
    return np.vstack((row, col)).squeeze().astype(int)


def map_correlation(grid_map, res, Y_io, scan_range):
    xdim = grid_map.shape[0]
    ydim = grid_map.shape[1]
    n = np.ceil(scan_range/res)
    x_scan = y_scan = np.arange(-n, n + 1).astype(int)
    n_xs = x_scan.size
    n_ys = y_scan.size
    corr = np.zeros((n_xs, n_ys))
    for jy in range(0, n_ys):
        iy = Y_io[1, :] + y_scan[jy]
        for jx in range(0, n_xs):
            ix = Y_io[0, :] + x_scan[jx]
            valid = np.logical_and(np.logical_and((iy >= 0), (iy < ydim)),
                                   np.logical_and((ix >= 0), (ix < xdim)))
            corr[jx, jy] = np.sum(grid_map[ix[valid], iy[valid]])
    return corr


def generate_video(png_dir):
    print('Generating video...')
    i = 0
    images = [[]] * 99999  # at most 99,999 frames
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images[i] = imageio.imread(file_path)
            i += 1
    imageio.mimsave(os.path.join(png_dir, 'result.mp4'), images[:i])
    print('Done.')


