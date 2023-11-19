import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def qtable_heatmap(qtable):
    # plt.figure(figsize=(20,25))
    if isinstance(qtable, np.ndarray):
        qtable = np.array(qtable)
    
    sns.heatmap(np.array(qtable), xticklabels=['up', 'right', 'down', 'left'])

def agent_decision(qtable):
    states_n, actions_n = len(qtable), len(qtable[0])
    rows = cols = int(np.sqrt(states_n))
    decision_mat = [[0 for _ in range(cols)] for _ in range(rows)]
    decisions = {
        0: 'up',
        1: 'right',
        2: 'down',
        3: 'left',
    }

    state = 0
    for i in range(rows):
        for j in range(cols):
            state = i * rows + j
            action = np.argmax(qtable[state])
            decision_mat[i][j] = decisions[action]
    
    return decision_mat