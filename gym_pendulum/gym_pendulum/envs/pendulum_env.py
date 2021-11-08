import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import quad
import tensorflow as tf
# set random seed for reproducibility





class PendulumEnvAdjusted(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=9.8):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.01
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.mu = 0.01
        self.viewer = None
        self.angle = None
        self.velocity = None
        self.both = None
        self.est_mass = 0.9

        low = np.array([0, -self.max_speed], dtype=np.float32)
        high = np.array([2*np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.action_space_est = spaces.Box(
            low=-2, high=2, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.angle_noise=None
        self.vel_noise=None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_angle_noise(self, angle_noise):
        self.angle_noise = angle_noise

    def set_vel_noise(self, vel_noise):
        self.vel_noise = vel_noise


    def step(self, u):
        th, thdot = self.state  # th := theta
        est_th, est_thdot = self.est_state

        g = self.g
        l = self.l
        mu = self.mu
        dt = self.dt

        #model estimation
        m = self.m
        param = self.est_mass+ u[4]
        m_est=param

        u_t = u[2]
        u_t = np.clip(u_t, -self.max_torque, self.max_torque)       
        self.last_u = u_t # for rendering
        a = np.array([u[0], u[1]]) # error comp. pars
        
        # model based on true states
        newthdot = (
            thdot
            + ((-1* g / ( 1*l) * np.sin(th +np.pi) - (mu / (m * l ** 2)) * thdot) + (1/ (m * l **2)) * u_t) * dt  + self.vel_noise
        )
        newth = th + newthdot * dt  + self.angle_noise
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        #update estimated states
        est_newthdot = (
            est_thdot + 
            ((-1* g / (1*l) * np.sin(est_th+np.pi) - (mu / (m_est * l ** 2)) * est_thdot) + (1 / (m_est * l **2)) * u_t) * dt + a[1]
        )
        est_newth = est_th + est_newthdot * dt + a[0]  
        est_newthdot = np.clip(est_newthdot, -self.max_speed, self.max_speed)

        # set measurements based on inputs
        if self.angle==True:
            y_th = u[3]
            y_est_th = est_th
        elif self.velocity == True:
            y_th = u[3]
            y_est_th = est_thdot
        else: 
            y_th1 = u[3][0]
            y_est_th1 = est_th
            y_th2 = u[3][1]
            y_est_th2 = est_thdot

            y_th = [y_th1, y_th2]
            y_est_th = [y_est_th1, y_est_th2]

        #compute rewards
        if self.angle == True or self.velocity == True:
            reward = -(y_th - y_est_th) ** 2 
        else: 
            reward = - ((y_th[0] - y_est_th[0]) ** 2 + (y_th[1] - y_est_th[1]) ** 2)/2 

        reward_pend =  -1 * (angle_normalize(est_th) ** 2 + 0.1 * est_thdot ** 2 + 0.001 * (u_t ** 2))

        reward_est = np.squeeze(reward)
        reward_pend =np.squeeze(reward_pend)

        #update state and est state
        self.state = np.array([newth, newthdot], dtype=np.float64)
        self.est_state = np.array([est_newth, est_newthdot], dtype=np.float64)

        #returns
        return_state = [self.state, self.est_state, param]
        reward = [reward_est, reward_pend]
        
        return return_state, reward, False, {}

    def set_angle(self, angle):
        if angle == "angle":
            self.angle=True
        if angle == "velocity":
            self.velocity=True
        if angle=="both":
            self.both = True

        print(self.angle, self.velocity, self.both)

    def reset(self):
        low = np.array([-np.pi, -0.5*np.pi]) 
        high = np.array([np.pi, 0.5*np.pi]) 
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def reset_est(self):
        theta, thetadot = self.state 
        low = np.array([-np.pi + theta, -np.pi+ thetadot])
        high = np.array([np.pi + theta, np.pi + thetadot])
        self.est_state = self.np_random.uniform(low=low, high=high)
        return self._get_est_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot], dtype=np.float64)

    def _get_est_obs(self):
        theta, thetadot = self.est_state
        return np.array([theta, thetadot], dtype=np.float64)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod2 = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            rod2.set_color(0.2, 0.5, 0.5)
            self.pole_transform = rendering.Transform()
            self.pole_transform2 = rendering.Transform()
            rod.add_attr(self.pole_transform)
            rod2.add_attr(self.pole_transform2)
            self.viewer.add_geom(rod)
            self.viewer.add_geom(rod2)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0]+1/2*np.pi)
        self.pole_transform2.set_rotation(self.est_state[0]+1/2*np.pi)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi



