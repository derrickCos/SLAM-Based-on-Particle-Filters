import copy
import numpy as np
import matplotlib.pyplot as plt

UNKNOWN = 0
WALL    = 2
FREE    = 1

class Map(object):
    def __init__(self, xmin=-5, ymin=-15, xmax=15, ymax=10, res=0.05, figsize=(8, 8)):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.res = res
        self.map, self.log_odds = self._init_map_and_odds(xmin, ymin, xmax, ymax, res)
        self.fig, self.ax = plt.subplots(figsize=figsize)


    def update(self, lidar_coords, state):

        if lidar_coords.size == 0:
            return

        # Mapping-b) transform points into world frame (Y_wo in pixels)
        wpb = np.array([state[0]]).T
        wRb = np.array([[np.cos(state[1]), -np.sin(state[1])],
                        [np.sin(state[1]), np.cos(state[1])]])
        bpl = np.array([[0, 0.015935]]).T
        Y_wo = np.matmul(wRb, lidar_coords + bpl) + wpb
        self._check_and_expand_map(Y_wo)
        # transform into pixels
        Y_wo = self._meter_to_pixel(Y_wo, self.xmin, self.ymax, self.res)

        # Mapping-c) Use bresenham2D to find free points (Y_wf in pixels)
        Y_wf = np.empty((2,0))
        wpb = self._meter_to_pixel(wpb, self.xmin, self.ymax, self.res)
        for i in range(Y_wo.shape[1]):
            Y_wf = np.hstack((Y_wf, self._bresenham2D(Y_wo[0, i], Y_wo[1, i], wpb[0], wpb[1])))
        # remove overlapping points
        Y_wf = np.unique(Y_wf, axis=1).astype(np.int_)

        # Mapping-d) increase/decrease log-odds
        trust = 0.8
        log_t = np.log(trust/(1 - trust))
        self.log_odds[Y_wo[0], Y_wo[1]] = self.log_odds[Y_wo[0], Y_wo[1]] + log_t
        self.log_odds[Y_wf[0], Y_wf[1]] = self.log_odds[Y_wf[0], Y_wf[1]] - log_t
        self.map[self.log_odds < 0] = FREE
        self.map[self.log_odds > 0] = WALL


    def show(self, time, trajectory, theta):
        self.ax.clear()
        self.ax.imshow(self.map, interpolation='none', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        self.ax.plot(trajectory[0][:-1], trajectory[1][:-1], 'r.', markersize=3)
        self.ax.plot(trajectory[0][-1], trajectory[1][-1], 'r', marker=(3, 0, theta/np.pi*180 - 90), markersize=5)
        self.ax.set_title('Time: %.3f s' % time)
        self.ax.set_xlabel('x / m')
        self.ax.set_ylabel('y / m')

    @staticmethod
    def show_particles(axe, particles):
        for p in particles:
            axe.plot(p.state[0][0], p.state[0][1], 'b.', markersize=2)


    @staticmethod
    def _init_map_and_odds(xmin, ymin, xmax, ymax, res):
        init_map = np.zeros((np.ceil((ymax - ymin)/res + 1).astype(np.int_),
                             np.ceil((xmax - xmin)/res + 1).astype(np.int_))).astype(np.int_) * UNKNOWN
        init_log_odds = np.zeros(np.shape(init_map))
        return init_map, init_log_odds


    @staticmethod
    def _meter_to_pixel(coord, xmin, ymax, res):
        # coord: size should be (2, *)
        coord_col = np.ceil((coord[0] - xmin)/res)
        coord_row = np.ceil((ymax- coord[1])/res)
        return np.squeeze(np.vstack((coord_row, coord_col))).astype(np.int_)


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
        return np.vstack((x, y))[:, 1:]


    def _check_and_expand_map(self, points):
        xmin_pre = self.xmin
        ymax_pre = self.ymax
        EXTEND = False
        if (points[0] < self.xmin).any():
            EXTEND = True
            xmin_pre = self.xmin
            self.xmin = xmin_pre * 2
        elif (points[0] > self.xmax).any():
            EXTEND = True
            xmax_pre = self.xmax
            self.xmax = xmax_pre * 2
        elif (points[1] < self.ymin).any():
            EXTEND = True
            ymin_pre = self.ymin
            self.ymin = ymin_pre * 2
        elif (points[1] > self.ymax).any():
            EXTEND = True
            ymax_pre = self.ymax
            self.ymax = ymax_pre * 2
        if EXTEND:
            map_pre = self.map
            log_odds_pre = self.log_odds
            self.map, self.log_odds = self._init_map_and_odds(self.xmin, self.ymin, self.xmax, self.ymax, self.res)
            # copy previous data into new ones
            coord = self._meter_to_pixel(np.array([xmin_pre, ymax_pre]), self.xmin, self.ymax, self.res)
            row_pre, col_pre = map_pre.shape
            self.map[coord[0]:coord[0] + row_pre, coord[1]:coord[1] + col_pre] = map_pre
            self.log_odds[coord[0]:coord[0] + row_pre, coord[1]:coord[1] + col_pre] = log_odds_pre





