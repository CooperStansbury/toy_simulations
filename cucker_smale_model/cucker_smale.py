
import pycxsimulator
from pylab import *
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csgraph
import random
import pandas as pd
import math


def initialize():
    global state, next_state, epoch
    state = np.random.uniform(-30, 30, (N_BIRDS, 3)) # pos x, pos y, velocity
    next_state = np.zeros((N_BIRDS, 3))
    epoch = 0


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
    
    
    angle = math.radians(bird[2] * 1/bird_dist)
    n = rotate(centroid, bird[0:2], angle)
    
    noise = np.random.normal(NOISE_M, NOISE_STD, 2)
    
    new_x = n[0] + noise[0]
    new_y = n[1] + noise[1]
    
    new_bird = np.array((new_x, new_y, bird[2]))
    return new_bird


def update():
   global state, next_state, epoch
   next_state = state.copy()
   
   epoch += 1
   
   for i, bird in enumerate(state):
       next_state[i] = move(i, state)
       next_state[i, 2] = update_velocities(i, state)
       
       row = {
           "bird" : i,
           "x_position" : next_state[i, 0] ,
           "y_position": next_state[i, 1],
           "velocity": next_state[i, 2], 
           "epoch" : epoch
       }
       
       RESULTS.append(row)
   
   state = next_state.copy()

    
def observe():
    global state, next_state
    cla()
    plt.scatter(state[:, 0], 
                state[:, 1], 
                c=state[:, 2],
                s=40,
                edgecolor='black',
                alpha=0.7, 
                cmap='plasma')
    plt.xlim(-PLOT_LIM,PLOT_LIM)
    plt.ylim(-PLOT_LIM,PLOT_LIM)
    plt.show()


if __name__ == '__main__':

    PLOT_LIM = 100
    N_BIRDS= 50
    TIMESTEPS = 10
    NOISE_M = 0.5
    NOISE_STD = 1
    
    
    K = 0.1
    sigma = 0.1
    beta = 0.6
    
    RESULTS = []
    
    # # manual running of experiment
    # initialize()
    # for i in range(TIMESTEPS):
    #     update()
        
    # # results formatting    
    # results = pd.DataFrame(RESULTS)

    
    # for bird in results['bird'].unique():
    #     tmp = results[results['bird'] == bird]
    #     plt.plot(tmp['epoch'], tmp['velocity'], lw=1, alpha=0.5, c='C0')
    
    # plt.title('Velocity over Time')
    # plt.show()
    
    

    pycxsimulator.GUI().start(func=[initialize, observe, update])