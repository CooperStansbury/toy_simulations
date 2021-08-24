
from seaborn import palettes
import pycxsimulator
from pylab import *
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.sparse import csgraph
import random
import pandas as pd
import math
from random import choice


def create_wound(state, wound_width, plot_limit):
    """A function to create the wound """
    midpoint = plot_limit // 2
    wound_upper_bound =  (plot_limit * wound_width)
    wound_lower_bound = -1 * (plot_limit * wound_width)
    state = state[(state[:,1] > wound_upper_bound) | (state[:,1] < wound_lower_bound)]
    return state
    

def initialize():
    global state, next_state, epoch
    
    epoch = 0

    state = np.random.uniform(-PLOT_LIMIT, PLOT_LIMIT, (N_CELLS, 2)) 
    state = create_wound(state, WOUND_WIDTH, PLOT_LIMIT)
    
    state = pd.DataFrame(state, columns=['x_position', 'y_position'])
    
    # create cell cycle column
    g1 = ['G1'] * int(PROP_G1 * len(state))
    s = ['S'] * int(PROP_S * len(state))
    g2 = ['G2'] * (len(state) - (len(g1) + len(s)))
    phase = g1 + s + g2
    state['phase'] = phase
    
    # generate ages (hours) by cell cycle
    state['age'] = 1
    
    state.loc[(state['phase'] == 'G1'), 'age'] = np.random.randint(1, 11)
    state.loc[(state['phase'] == 'S'), 'age'] = np.random.randint(12, 17)
    state.loc[(state['phase'] == 'G2'), 'age'] = np.random.randint(18, 24)
    next_state = state.copy()
    epoch = 0


def move(cell):
    """A simple function to move towards the midline (y=0) with a decaying
    velocity inversly proportional to it's distance  """
    
    if cell['y_position'] < 0:
        dist2move = (1 / np.abs(cell['y_position'])) * 5 * np.sqrt(np.abs(cell['y_position']) )
        new_y = cell['y_position'] + dist2move 
    else: 
        dist2move = (1 / np.abs(cell['y_position'])) * 5 * np.sqrt(np.abs(cell['y_position']) )
        new_y = cell['y_position'] - dist2move
    return new_y
        
    
    
    


def update():
   global state, next_state, epoch
   next_state = state.copy()
   
   for i, cell in state.iterrows():
       next_state.at[i, 'y_position'] = move(cell)
    
       
    #    break
       
    #    row = {
    #        "bird" : i,
    #        "x_position" : next_state[i, 0] ,
    #        "y_position": next_state[i, 1],
    #        "velocity": next_state[i, 2], 
    #        "epoch" : epoch
    #    }
       
    #    RESULTS.append(row)
   
   state = next_state.copy()

    
def observe():
    global state, next_state
    cla()
    sns.scatterplot(data=state,
                    x='x_position',
                    y='y_position',
                    hue='phase',
                    s=40,
                    edgecolor='black',
                    alpha=0.7, 
                    palette=['C2', 'C3', 'C0'])
    plt.xlim(-PLOT_LIMIT, PLOT_LIMIT)
    plt.ylim(-PLOT_LIMIT, PLOT_LIMIT)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    PLOT_LIMIT = 100
    N_CELLS = 300
    HOURS = 3
    WOUND_WIDTH = 0.4 # expressed as percentage of y axis
    
    # these must add to 1
    PROP_G1 = 0.4
    PROP_S = 0.3
    PROP_G2 = 0.3
    
    RESULTS = []
    
    # # manual running of experiment
    # initialize()
    # # # observe()
    # for i in range(HOURS):
    #     update()
        
    # # results formatting    
    # results = pd.DataFrame(RESULTS)

    
    # for bird in results['bird'].unique():
    #     tmp = results[results['bird'] == bird]
    #     plt.plot(tmp['epoch'], tmp['velocity'], lw=1, alpha=0.5, c='C0')
    
    # plt.title('Velocity over Time')
    # plt.show()
    
    

    pycxsimulator.GUI().start(func=[initialize, observe, update])