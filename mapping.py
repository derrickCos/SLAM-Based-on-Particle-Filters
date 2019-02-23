import numpy as np
import matplotlib.pyplot as plt
from utils import world_to_image, to_homo, to_non_homo, rc_to_uv, WALL, FREE, UNKNOWN



class Map(object):
    def __init__(self, xmin=-5, ymin=-15, xmax=15, ymax=10, res=0.05, figsize=(16, 8)):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.res = res
        self.log_odd_prior = 0
        self.grid_map, self.log_odds, self.texture = None, None, None
        self._init_maps()
        _, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize)


    def update_map(self, lidar_coords, T_wl):

        # skip if no effect lidar scan points
        if lidar_coords.size == 0:
            return

        # Mapping-b) transform points into image frame (Y_io in image frame)
        Y_wo = to_non_homo(np.matmul(T_wl, to_homo(lidar_coords)))
        self._check_and_expand_maps(Y_wo)
        Y_io = world_to_image(Y_wo, self.xmin, self.ymax, self.res)
        # remove overlapping points
        # Y_io = np.unique(Y_io, axis=1)

        # Mapping-c) Use bresenham2D to find free points (Y_if in image frame)
        Y_if = np.empty((2,0))
        p_wb = T_wl[0:2, 2]
        p_ib = world_to_image(p_wb, self.xmin, self.ymax, self.res)
        for i in range(Y_io.shape[1]):
            Y_if = np.hstack((Y_if, self._bresenham2D(Y_io[0, i], Y_io[1, i], p_ib[0], p_ib[1])))
        # remove overlapping points
        Y_if = Y_if.astype(int)
        # Y_if = np.unique(Y_if, axis=1).astype(int)

        # Mapping-d) increase/decrease log-odds
        trust = 0.8
        log_t = np.log(trust/(1 - trust))
        # notice that occupied points are included in Y_if, so we add 2*log_t
        self.log_odds[Y_io[0], Y_io[1]] = self.log_odds[Y_io[0], Y_io[1]] + 2*log_t
        self.log_odds[Y_if[0], Y_if[1]] = self.log_odds[Y_if[0], Y_if[1]] - log_t
        self.grid_map[self.log_odds < self.log_odd_prior] = FREE
        self.grid_map[self.log_odds > self.log_odd_prior] = WALL


    def update_texture(self, rgb, disp, K_oi, T_wo, floor_threshold):
        I, J = disp.shape
        dd = -0.00304 * disp + 3.31
        depth = 1.03 / dd
        j, i = np.meshgrid(np.arange(J), np.arange(I))
        # floor coords
        Y_im = np.vstack([i.reshape(-1), j.reshape(-1)]).astype(int)
        Y_o = np.matmul(K_oi, to_homo(rc_to_uv(Y_im, I))) * depth.reshape(-1)
        Y_w = to_non_homo(np.matmul(T_wo, to_homo(Y_o)))

        # floor thresholing
        floor_index = np.abs(Y_w[2]) < floor_threshold
        floor_w = Y_w[:2, floor_index]
        if floor_w.size == 0:
            return
        floor_i, indices= np.unique(world_to_image(floor_w, self.xmin, self.ymax, self.res), axis=1, return_index=True)
        floor_im = Y_im[:, floor_index][:, indices]


        depth_i, depth_j = floor_im[0], floor_im[1]
        rgb_i = np.round((depth_i * 526.37 - 7877.07 * dd[depth_i, depth_j] + 19276.0) / 585.051).astype(int)
        rgb_j = np.round((depth_j * 526.37 + 16662.0) / 585.051).astype(int)
        weight = self.texture[floor_i[0], floor_i[1], 3]
        self.texture[floor_i[0], floor_i[1], :3] = ((self.texture[floor_i[0], floor_i[1], :3].T*weight +
                                                    rgb[rgb_i, rgb_j].T)/(weight + 1)).T
        self.texture[floor_i[0], floor_i[1], 3] += 1


    def show(self, time, trajectory, theta):
        self.ax1.clear()
        self.ax1.imshow(self.grid_map, interpolation='none', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        self.ax1.plot(trajectory[0][:-1], trajectory[1][:-1], 'r.', markersize=3)
        self.ax1.plot(trajectory[0][-1], trajectory[1][-1], 'r', marker=(3, 0, theta/np.pi*180 - 90), markersize=5)
        self.ax1.set_title('Time: %.3f s' % time)
        self.ax1.set_xlabel('x / m')
        self.ax1.set_ylabel('y / m')
        self.ax2.clear()
        self.ax2.imshow(self.texture[:, :, :3], extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        self.ax2.plot(trajectory[0][-1], trajectory[1][-1], 'r', marker=(3, 0, theta/np.pi*180 - 90), markersize=5)
        self.ax2.set_title('Texture mapping')


    @staticmethod
    def show_particles(ax, particles):
        for p in particles:
            ax.plot(p.state[0][0], p.state[0][1], 'b.', markersize=2)


    def _init_maps(self):
        self.grid_map = np.zeros((np.ceil((self.ymax - self.ymin)/self.res + 1).astype(np.int),
                             np.ceil((self.xmax - self.xmin)/self.res + 1).astype(np.int))).astype(int) * UNKNOWN
        self.log_odds = np.zeros(self.grid_map.shape) + self.log_odd_prior
        self.texture = np.zeros(self.grid_map.shape + (4,))


    @staticmethod
    def _bresenham2D(sx, sy, ex, ey):
        '''
        Bresenham's ray tracing algorithm in 2D.
        Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
        Return:
            do no include the start point of ray
        '''
        sx = int(np.round(sx))
        sy = int(np.round(sy))
        ex = int(np.round(ex))
        ey = int(np.round(ey))
        dx = abs(ex - sx)
        dy = abs(ey - sy)
        steep = abs(dy) > abs(dx)
        if steep:
            dx, dy = dy, dx  # swap

        if dy == 0:
            q = np.zeros((dx + 1, 1))
        else:
            q = np.append(0, np.greater_equal(
                np.diff(np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)), 0))
        if steep:
            if sy <= ey:
                y = np.arange(sy, ey + 1)
            else:
                y = np.arange(sy, ey - 1, -1)
            if sx <= ex:
                x = sx + np.cumsum(q)
            else:
                x = sx - np.cumsum(q)
        else:
            if sx <= ex:
                x = np.arange(sx, ex + 1)
            else:
                x = np.arange(sx, ex - 1, -1)
            if sy <= ey:
                y = sy + np.cumsum(q)
            else:
                y = sy - np.cumsum(q)
        # remove start point
        return np.vstack((x, y))


    def _check_and_expand_maps(self, coords):
        xmin_pre = self.xmin
        ymax_pre = self.ymax
        EXTEND = False
        if (coords[0] < self.xmin).any():
            EXTEND = True
            xmin_pre = self.xmin
            self.xmin = xmin_pre * 2
        elif (coords[0] > self.xmax).any():
            EXTEND = True
            xmax_pre = self.xmax
            self.xmax = xmax_pre * 2
        elif (coords[1] < self.ymin).any():
            EXTEND = True
            ymin_pre = self.ymin
            self.ymin = ymin_pre * 2
        elif (coords[1] > self.ymax).any():
            EXTEND = True
            ymax_pre = self.ymax
            self.ymax = ymax_pre * 2
        if EXTEND:
            map_pre = self.grid_map
            log_odds_pre = self.log_odds
            texture_pre = self.texture
            self._init_maps()
            # copy previous data into new ones
            coord = world_to_image(np.array([xmin_pre, ymax_pre]), self.xmin, self.ymax, self.res)
            row_pre, col_pre = map_pre.shape
            self.grid_map[coord[0]:coord[0] + row_pre, coord[1]:coord[1] + col_pre] = map_pre
            self.log_odds[coord[0]:coord[0] + row_pre, coord[1]:coord[1] + col_pre] = log_odds_pre
            self.texture[coord[0]:coord[0] + row_pre, coord[1]:coord[1] + col_pre] = texture_pre

