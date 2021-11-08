
import pickle
import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
from tensorflow.keras import layers, Model
from tensorflow import keras
from scipy import signal
import matplotlib.pyplot as plt
from ddpg import *
from ppo2 import *
import gym_pendulum


def get_next_meas(th, thdot, u_t, angle_noise, vel_noise, R):
    dt = 0.01
    g = 9.8
    l = 1
    m = 1
    mu = 0.01

    noise_th = np.random.normal(0, R*0.01)
    noise_thdot = np.random.normal(0, R*0.01)

    newthdot = (
            thdot
            + ((-1* g / ( 1*l) * np.sin(th +np.pi) - (mu / (m * l ** 2)) * thdot) + (1/ (m * l **2)) * u_t) * dt + noise_thdot
        )+vel_noise + noise_thdot
    newth = th + newthdot * dt  + angle_noise +noise_th

    return newth, newthdot

def get_next_meas_angle(th, thdot, u_t, angle_noise, vel_noise, R):
    dt = 0.01
    g = 9.8
    l = 1
    m = 1
    mu = 0.01

    noise_th = np.random.normal(0, R*0.01)
    # print(R)

    newthdot = (
            thdot
            + ((-1* g / ( 1*l) * np.sin(th +np.pi) - (mu / (m * l ** 2)) * thdot) + (1/ (m * l **2)) * u_t) * dt 
        ) +vel_noise
    newth = th + newthdot * dt + angle_noise + noise_th

    return newth

def get_next_meas_vel(th, thdot, u_t, angle_noise, vel_noise, R):
    dt = 0.01
    g = 9.8
    l = 1
    m = 1
    mu = 0.01

    noise_thdot = np.random.normal(0, R*0.01)

    newthdot = (
            thdot
            + ((-1* g / ( 1*l) * np.sin(th +np.pi) - (mu / (m * l ** 2)) * thdot) + (1/ (m * l **2)) * u_t) * dt + noise_thdot
        ) +vel_noise +noise_thdot
    newth = th + newthdot * dt + angle_noise

    return newthdot

