
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
    global state, next_state, time
    x_pos = np.random.uniform(-30, 30, (N_CELLS)) 
    y_pos = np.random.uniform(-30, 30, (N_CELLS)) 
    phase = np.random.randint(0, 25, (N_CELLS))
    state = np.column_stack((x_pos, y_pos, phase))

    next_state = np.zeros((N_CELLS, 3))
    time = 0


def move(cell):
    noise = np.random.normal(0, NOISE_STD, 2)
    new_x = cell[0] + noise[0]
    new_y = cell[1] + noise[1]
    return np.array((new_x, new_y, cell[2]))
    

def divide(cell):
    d1 = move(cell)
    d2 = move(cell)
    
    d1[2] = 0
    d2[2] = 0 
    return d1, d2
    
def update():
   global state, next_state, time
   next_state = state.copy()
   
   time += 1
   
   for i, cell in enumerate(state):
       next_state[i] = move(state[i])
       next_state[i, 2] += 1
       
       if next_state[i, 2] > 23:
           d1, d2 = divide(state[i])
           
           # add daughters
           next_state = np.vstack((next_state, d1))
           next_state = np.vstack((next_state, d2))
           
           # remove parent cell 
           np.delete(next_state, i, 0)
           
           
   row = {
       'n_cells': len(next_state)
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
                cmap='RdYlGn_r')
    plt.show()


if __name__ == '__main__':

    N_CELLS= 10
    TIMESTEPS = 10
    NOISE_STD = 1
    
    
    RESULTS = []
    
    # manual running of experiment
    initialize()
    for i in range(TIMESTEPS):
        update()
        
    # results formatting    
    results = pd.DataFrame(RESULTS)
    
    plt.plot(list(range(time)), results['n_cells'])
    plt.show()

    # pycxsimulator.GUI().start(func=[initialize, observe, update])