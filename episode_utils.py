import numpy as np



def filter_substeps(history):
    '''Returns a history where substeps are filtered out'''
    current_step = 0
    previous_step = 0
    new_history = []
    for i in range(len(history)):
        step = history[i]
        current_step = step
        if not previous_step:
            previous_step = current_step
            new_history.append(current_step)
            continue
        #the assumption that the same beacon postions means the same episode
        if np.array_equal(current_step['neutral_x'], previous_step['neutral_x']) and \
                np.array_equal(current_step['neutral_y'], previous_step['neutral_y']) and \
                np.array_equal(current_step['enemy_x'], previous_step['enemy_x']) and \
                np.array_equal(current_step['enemy_y'], previous_step['enemy_y']):
            continue
        else:
            previous_step = current_step
            new_history.append(current_step)

        previous_step = current_step




    return new_history