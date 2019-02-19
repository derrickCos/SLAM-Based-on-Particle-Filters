import copy
import numpy as np
from utils import add_noise



class Robot(object):
    def __init__(self, state, N, mode='delta'):
        self.N = N
        self.particles = Particle.create_particles(state, N, mode=mode)
        self.state = None
        self.trajectory = np.empty((2, 0))
        self._update_state()


    def advance_by(self, encoder_counts, yaw, dt, noisy=False):
        if dt == 0:
            return
        for p in self.particles:
            p.advance_by(encoder_counts, yaw, dt, noisy=noisy)
        self._update_state()


    def update_particles(self, lidar_coords, mapping, res, xmin, ymax):
        new_weights = np.zeros(self.N)
        for i in range(self.N):
            state = self.particles[i].state
            wpb = np.array([state[0]]).T
            wRb = np.array([[np.cos(state[1]), -np.sin(state[1])],
                            [np.sin(state[1]), np.cos(state[1])]])
            bpl = np.array([[0, 0.015935]]).T
            Y_wo = np.matmul(wRb, lidar_coords + bpl) + wpb
            x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
            y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
            new_weights[i] = np.max(self._mapCorrelation((mapping > 0).astype(np.int_), res,
                                                         xmin, ymax, Y_wo, x_range, y_range))
        if (new_weights > 100).any():   # handles overflow
            new_weights = new_weights - np.max(new_weights) + 100
            # new_weights = new_weights / np.max(new_weights)
        for i in range(self.N):
            self.particles[i].w = self.particles[i].w * np.exp(new_weights[i])
        # nu = sum(p.w for p in self.particles)
        # w_squared_sum = 0
        # for p in self.particles:
        #     p.w = p.w/nu
        #     w_squared_sum += p.w**2
        # N_eff = 1/w_squared_sum
        # if N_eff < self.N/3*2:
        self._stratified_resampling()
        self._update_state()


    def _stratified_resampling(self):
        particles = copy.deepcopy(self.particles)
        j = 0
        accum = self.particles[0].w
        for i in range(self.N):
            u = np.random.rand(1)/self.N
            beta = u + (i - 1)/self.N
            while beta > accum:
                j += 1
                accum += particles[j].w
            self.particles[i] = Particle.create_particles(particles[j].state, 1)[0]


    @staticmethod
    def _mapCorrelation(im, res, xmin, ymax, vp, xs, ys):
        '''
        INPUT
        im              the map
        x_im, y_im      physical x,y positions of the grid map cells
        vp[0:2, :]      occupied x,y positions from range sensor (in physical unit)
        xs,ys           physical x,y,positions you want to evaluate "correlation"

        OUTPUT
        c               sum of the cell values of all the positions hit by range sensor
        '''
        nx = im.shape[0]
        ny = im.shape[1]
        nxs = xs.size
        nys = ys.size
        cpr = np.zeros((nxs, nys))
        for jy in range(0, nys):
            y1 = vp[1, :] + ys[jy]  # 1 x 1076
            iy = np.int16(np.round((ymax - y1) / res))
            for jx in range(0, nxs):
                x1 = vp[0, :] + xs[jx]  # 1 x 1076
                ix = np.int16(np.round((x1 - xmin) / res))
                valid = np.logical_and(np.logical_and((iy >= 0), (iy < ny)),
                                       np.logical_and((ix >= 0), (ix < nx)))
                cpr[jx, jy] = np.sum(im[ix[valid], iy[valid]])
        return cpr


    def _update_state(self):
        d = {}
        for p in self.particles:
            d[p] = d.get(p, 0) + p.w
        weights = list(d.values())
        particles = list(d.keys())
        self.state = particles[weights.index(max(weights))].state
        self.trajectory = np.hstack((self.trajectory, np.array([self.state[0]]).T))



class Particle(object):
    def __init__(self, state, weight=1, noisy=False):
        (x, y), theta = state
        if noisy:
            self.state = [(add_noise('coord', x), add_noise('coord', y)), add_noise('theta', theta)]
        else:
            self.state = [(x, y), theta]
        self.w = weight


    @classmethod
    def create_particles(cls, state, count, mode='delta'):
        if mode == 'delta':
            return [cls(state) for _ in range(0, count)]
        elif mode == 'uniform':
            (x, y), theta = state
            particles = [[]] * count
            for i in range(count):
                x0 = x + np.random.uniform(-.1, .1)
                y0 = y + np.random.uniform(-.1, .1)
                theta0 = theta
                particles[i] = cls([(x0, y0), theta0])
            return particles
        else:
            raise ValueError('Incorrect mode type.')


    def advance_by(self, encoder_counts, yaw, dt, noisy=False):
        (x0, y0), theta0 = self.state
        fr, fl, rr, rl = encoder_counts
        s_r = (fr + rr) / 2 * 0.0022
        s_l = (fl + rl) / 2 * 0.0022
        v_r = s_r / dt
        v_l = s_l / dt
        v = (v_r + v_l) / 2
        dtheta = yaw * dt
        if noisy:
            v = add_noise('speed', v)
            dtheta = add_noise('theta', dtheta)
        theta = theta0 + dtheta
        x = x0 + v*dt*np.sinc(dtheta/2/np.pi)*np.cos(theta)
        y = y0 + v*dt*np.sinc(dtheta/2/np.pi)*np.sin(theta)
        self.state = (x, y), theta



