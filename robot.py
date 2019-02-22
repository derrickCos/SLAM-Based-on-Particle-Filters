import copy
import numpy as np
from utils import *


class Robot(object):
    def __init__(self, state, N):
        ## Particles
        self.N = N
        self.particles = Particle.create_particles(N - 1, state)
        self.particles.append(Particle.create_particles(1, state)[0])

        ## Kinect parameters
        # extrinsic matrix
        roll, pitch, yaw = 0, 0.36, 0.021
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_bc = np.matmul(np.matmul(R_z, R_y), R_x)
        p_bc = np.array([0.18, 0.005, 0.36])
        R_oc = np.array([[0, -1, 0],
                         [0, 0, -1],
                         [1, 0, 0]])
        T_ob = np.vstack((np.hstack((np.matmul(R_oc, R_bc.T), -np.matmul(R_oc, np.matmul(R_bc.T, p_bc.reshape(3, 1))))),
                          np.array([[0, 0, 0, 1]])))
        self.T_bo = np.linalg.inv(T_ob)
        # intrisic matrix
        K_io = np.array([[585.05108211, 0, 242.94140713],
                      [0, 585.05108211, 315.83800193],
                      [0, 0, 1]])
        self.K_oi = np.linalg.inv(K_io)

        ## LIDAR parameters
        p_bl = np.array([0, 0.015935])
        R_bl = np.eye(2)
        self.T_bl = np.vstack((np.hstack((R_bl, p_bl.reshape(2, 1))),
                               np.array([[0, 0, 1]])))

        ## robot state and trajectory
        self.state = None
        self.T_wb = None
        self.trajectory = np.empty((2, 0))
        self._update_state()


    def advance_by(self, encoder_counts, yaw, dt, noisy=False):
        for i in range(self.N - 1):
            self.particles[i].advance_by(encoder_counts, yaw, dt, noisy=noisy)
        self.particles[-1].advance_by(encoder_counts, yaw, dt)


    def update_particles(self, lidar_coords, grid_map, res, xmin, ymax):
        grid_map = 2 * (grid_map == WALL) - 1      # convert to {-1, 1}
        new_weights = np.zeros(self.N)
        for i in range(self.N):
            p_wb = np.array(self.particles[i].state[0])
            R_wb = np.array([[np.cos(self.particles[i].state[1]), -np.sin(self.particles[i].state[1])],
                             [np.sin(self.particles[i].state[1]), np.cos(self.particles[i].state[1])]])
            self.T_wb = np.vstack((np.hstack((R_wb, p_wb.reshape(2, 1))), np.array([[0, 0, 1]])))
            T_wl = np.matmul(self.T_wb, self.T_bl)
            Y_wo = to_non_homo(np.matmul(T_wl, to_homo(lidar_coords)))
            self.particles[i].Y_io = np.unique(np.hstack((self.particles[i].Y_io,
                                                          world_to_image(Y_wo, xmin, ymax, res))), axis=1).astype(int)
            new_weights[i] = np.max(map_correlation(grid_map, res, self.particles[i].Y_io, scan_range=0.2))
        new_weights = new_weights - np.max(new_weights) + 50    # avoid overflow
        new_weights = np.exp(new_weights)/np.sum(np.exp(new_weights))
        for i in range(self.N):
            self.particles[i].w = self.particles[i].w * new_weights[i]
        self._stratified_resampling()
        self._update_state()


    def _stratified_resampling(self):
        particles = copy.deepcopy(self.particles)
        j = 0
        accum = self.particles[0].w
        for i in range(self.N):
            u = np.random.rand(1)/self.N
            beta = u + i/self.N
            while beta > accum:
                j += 1
                accum += particles[j].w
            self.particles[i] = Particle.create_particles(1, particles[j].state, particles[j].Y_io)[0]


    def _update_state(self):
        d = {}
        for p in self.particles:
            d[p.state] = d.get(p.state, 0) + p.w
        weights = list(d.values())
        states = list(d.keys())
        self.state = states[weights.index(max(weights))]
        p_wb = np.array(self.state[0])
        R_wb = np.array([[np.cos(self.state[1]), -np.sin(self.state[1])],
                         [np.sin(self.state[1]), np.cos(self.state[1])]])
        self.T_wb = np.vstack((np.hstack((R_wb, p_wb.reshape(2, 1))),
                               np.array([[0, 0, 1]])))
        self.trajectory = np.hstack((self.trajectory, p_wb.reshape(2, 1)))



class Particle(object):
    def __init__(self, state, Y_io, weight=1):
        self.state = state
        self.w = weight
        self.Y_io = Y_io


    @classmethod
    def create_particles(cls, count, state, Y_io=np.empty((2, 0))):
            return [cls(state, Y_io) for _ in range(0, count)]


    def advance_by(self, encoder_counts, yaw, dt, noisy=False):
        (x0, y0), theta0 = self.state
        fr, fl, rr, rl = encoder_counts
        s_r = (fr + rr) / 2 * 0.0022
        s_l = (fl + rl) / 2 * 0.0022
        v_r = s_r / dt
        v_l = s_l / dt
        v = (v_r + v_l) / 2
        if noisy:
            v = add_noise(v, sigma_speed)
            yaw = add_noise(yaw, sigma_yaw)
        dtheta = yaw * dt
        theta = theta0 + dtheta
        x = x0 + v*dt*np.sinc(dtheta/2/np.pi)*np.cos(theta)
        y = y0 + v*dt*np.sinc(dtheta/2/np.pi)*np.sin(theta)
        self.state = (x, y), theta



