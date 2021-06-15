
import pycxsimulator
from pylab import *
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csgraph
import random
import math


def initialize():
    global state, next_state
    state = np.random.uniform(-10, 10, (N_BIRDS, 3)) # pos x, pos y, velocity
    next_state = np.zeros((N_BIRDS, 3))

# def get_eta(a_ij):
#     """A function to return eta per the paper"""
#     return K / (sigma**2 + a_ij)**beta

# def get_Ax(state):
#     """function to compute the Laplacian at t """
#     Ax = np.zeros((N_BIRDS, N_BIRDS))
    
#     for i in range(N_BIRDS):
#         for j in range(N_BIRDS):
#             b1 = state[i, 0:2]
#             b2 = state[j, 0:2]
            
#             a_ij = np.linalg.norm(b1 - b2)
#             Ax[i, j] = get_eta(a_ij)
            
#     return Ax

  
# def update_velocities(i, state, Lx):
#     """compute velocity differential"""
#     new_vel = - (Lx[i] * state[:, 2])[i] + state[i, 2]
#     return new_vel


def update_velocities(i, state):
    """An update function for the ith bird"""
    
    velocity_diffs = []
    
    for j in range(len(state)):
        if not j == i:
            diff = state[i, 2] - state[j, 2]        
            dist = np.linalg.norm(state[i, 0:2] - state[j, 0:2])
            a_ij = K / (sigma**2 + dist)**beta
            weighted_diff = diff * a_ij
            
            velocity_diffs.append(weighted_diff)
            
    new_velocity = state[i, 2] + np.sum(velocity_diffs) 
    return new_velocity


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
 
            
def move(i, state):
    """a function to move  a bird"""
    
    bird = state[i]
    
    centroid = np.mean(state, axis=0)[0:2]
    
    dists = [np.linalg.norm(state[k, 0:2] - centroid) for k in range(len(state))]
    dists = [k/np.sum(dists) for k in dists]
    bird_dist = dists[i]
    
    
    angle = math.radians(bird[2] * bird_dist)
    n = rotate(centroid, bird[0:2], angle)
    
    noise = np.random.normal(NOISE_M, NOISE_STD, 2)
    
    new_x = n[0] + noise[0]
    new_y = n[1] + noise[1]
    
    new_bird = np.array((new_x, new_y, bird[2]))
    return new_bird


def update():
   global state, next_state
   next_state = state.copy()
   
   for i, bird in enumerate(state):
       next_state[i] = move(i, state)
       next_state[i, 2] = update_velocities(i, state)
   
   state = next_state.copy()

    
def observe():
    global state, next_state
    cla()
    plt.scatter(state[:, 0], 
                state[:, 1], 
                c=state[:, 2],
                s=20,
                edgecolor='black',
                alpha=0.7, 
                cmap='plasma')
    plt.xlim(-PLOT_LIM,PLOT_LIM)
    plt.ylim(-PLOT_LIM,PLOT_LIM)
    plt.show()


if __name__ == '__main__':

    PLOT_LIM = 100
    N_BIRDS= 50
    VELOCITY_COEFF = 0.05
    TIMESTEPS = 2
    NOISE_M = 0.5
    NOISE_STD = 1
    
    
    K = 0.1
    sigma = 0.1
    beta = 0.6
    
    # initialize()
    # for i in range(TIMESTEPS):
    #     update()
        
    # observe()
    
    

    pycxsimulator.GUI().start(func=[initialize, observe, update])