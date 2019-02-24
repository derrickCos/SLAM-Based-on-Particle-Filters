import os
import imageio
import numpy as np
from scipy.signal import butter, lfilter


# map label
UNKNOWN = 0
WALL    = 2
FREE    = 1


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


# def map_correlation(grid_map, res, Y_io, scan_range_xy, scan_range_w):
#     rdim, cdim = grid_map.shape
#     n = np.ceil(scan_range_xy/res)
#     r_scan = c_scan = np.arange(-n, n + 1).astype(int)
#     n_r = r_scan.size
#     n_c = c_scan.size
#     corr = np.zeros((n_r, n_c))
#     for ir in range(0, n_r):
#         r = Y_io[0] + r_scan[ir]
#         for ic in range(0, n_c):
#             c = Y_io[1] + c_scan[ic]
#             valid = np.logical_and(np.logical_and((r >= 0), (r < rdim)),
#                                    np.logical_and((c >= 0), (c < cdim)))
#             corr[ir, ic] = np.sum(grid_map[r[valid], c[valid]])
#     return corr


def map_correlation(grid_map, res, Y_io, scan_range_xy, scan_range_w):
    grid_map[0, 0] = 0
    rdim, cdim = grid_map.shape
    n_xy = np.ceil(scan_range_xy/res)
    # w_res = 0.001
    # w_scan = np.arange(-scan_range_w, scan_range_w + w_res, w_res)
    w_scan = np.array([0])
    corr = np.zeros(w_scan.size)
    for i in range(w_scan.size):
        w = w_scan[i]
        R_w = np.array([[np.cos(w), -np.sin(w)],
                        [np.sin(w), np.cos(w)]])
        Y_io_w = np.round(np.matmul(R_w, Y_io)).astype(int)
        r_scan = c_scan = np.arange(-n_xy, n_xy + 1).astype(int)
        n_scan = r_scan.size * c_scan.size
        Y_io_rep = np.repeat(Y_io_w[np.newaxis, :, :], n_scan, axis=0)
        r_s, c_s = np.meshgrid(r_scan, c_scan)
        scan = np.expand_dims(np.vstack((r_s.reshape(-1), c_s.reshape(-1))).T, axis=2)
        Y_scan = Y_io_rep + scan
        invalid = np.logical_or(np.logical_or(Y_scan[:, 0, :] < 0, Y_scan[:, 0, :] >= rdim),
                                np.logical_or(Y_scan[:, 1, :] < 0, Y_scan[:, 1, :] >= cdim))
        Y_scan[np.repeat(invalid[:, np.newaxis, :], 2, axis=1)] = 0
        corr[i] = np.max(np.sum(grid_map[Y_scan[:, 0], Y_scan[:, 1]], axis=1))
    # print(corr)
    return corr.max()


def load_and_process_data(dataset, texture_on):
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

    if texture_on:
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
        'encoder_v': np.zeros(len_stamps),
        'encoder_v_var': np.zeros(len_stamps),
        'lidar_update': [False] * len_stamps,
        'lidar_coords': [[]] * len_stamps,
        'imu_update': [False] * len_stamps,
        'imu_w': np.zeros(len_stamps),
        'imu_w_var': np.zeros(len_stamps),
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
            encoder_count = np.mean(encoder_counts[:, idx_l:idx_r],axis=1)
            fr, fl, rr, rl = encoder_count
            s_r = (fr + rr) / 2 * 0.0022
            s_l = (fl + rl) / 2 * 0.0022
            v_r = s_r / data['dt'][idx_t]
            v_l = s_l / data['dt'][idx_t]
            data['encoder_v'][idx_t] = (v_r + v_l) / 2
            idx_l = idx_r
        idx_t = idx_t + 1

        if idx_r >= len(encoder_stamps) or idx_t >= len(data['stamps']):
            break
    for i in range(len_stamps):
        data['encoder_v_var'][i] = np.var(data['encoder_v'][max(0, i - 3) : min(len_stamps, i + 3)])

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
    def low_pass_filtering(x, fs):
        cutoff = 10  # stop band: 10 Hz
        order = 10
        normal_cutoff = 2 * cutoff / fs
        b, a = butter(order, normal_cutoff)
        return lfilter(b, a, x)

    imu_yaw = imu_angular_velocity[2]
    imu_yaw_lp = low_pass_filtering(imu_yaw, 1 / np.mean(imu_stamps[1:] - imu_stamps[:-1]))
    idx_t = 0
    idx_l = 0
    while True:
        idx_r = idx_l
        while idx_r < len(imu_stamps) and imu_stamps[idx_r] <= data['stamps'][idx_t]:
            idx_r = idx_r + 1
        if idx_r != idx_l:
            data['imu_update'][idx_t] = True
            data['imu_w'][idx_t] = np.mean(imu_yaw_lp[idx_l:idx_r])
            idx_l = idx_r
        idx_t = idx_t + 1

        if idx_r >= len(imu_stamps) or idx_t >= len(data['stamps']):
            break
    for i in range(len_stamps):
        data['imu_w_var'][i] = np.var(data['imu_w'][max(0, i - 3) : min(len_stamps, i + 3)])

    # sync rgb
    if texture_on:
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
    if texture_on:
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


def check_and_rename(save_dir):
    if os.path.exists(save_dir):
        i = 1
        while os.path.exists(save_dir + '_' + str(i)):
            i += 1
        return save_dir + '_' + str(i)
    else:
        return save_dir



def generate_video(save_dir, format_list=['mp4', 'gif']):
    print('Generating video...')
    i = 0
    images = [[]] * 99999  # at most 99,999 frames
    for file_name in sorted(os.listdir(save_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(save_dir, file_name)
            images[i] = imageio.imread(file_path)
            i += 1
    for file_format in format_list:
        imageio.mimsave(os.path.join(save_dir, 'result.' + file_format), images[:i])
    print('Done. Video has been save to \'' + save_dir + '\'.')



