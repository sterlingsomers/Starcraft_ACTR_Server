import pickle
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import svm
import os
import episode_utils


input_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/data/','history_vectors.p')
output_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/', 'chunks_no_vectors.p')

minimum_history_length = 10
maximum_history_length = 10
vector_flag = True

history = pickle.load(open(input_filepath,"rb"))


#don't need every single step. Want first step
history = episode_utils.filter_substeps(history)


if not vector_flag:
    if len(history) < minimum_history_length:
        raise ValueError("History does not meet minimum length requirements")
    history = np.random.choice(history,maximum_history_length)

    chunks = []
    for case in history:
        ck = ['isa','decision','green',int(case['green']),
              'orange',int(case['orange']),'blocking',int(case['blocking']),
              'action', case['chosen_action']]#,
              #'vector',repr(case['vector'].tolist())]
              #'value_estimate',repr(case['value_estimate']),
              #'vector',repr(list(case['fc1']))]



        chunks.append(ck.copy())


    pickle.dump(chunks, open(output_filepath,"wb"))

else:
    green_only = 0
    orange_only = 0
    green_orange_non = 0
    green_orange_blocking = 0
    viables = []
    history.reverse()
    minimum_distance = 100
    step = 1
    max_cases = 5

    while minimum_distance >= 0:
        for case in history:
            number_same_cases = 0
            smallest_distance = 10000
            case_distance = 100000
            for viable in viables:
                if viable['green'] == case['green'] and \
                        viable['orange'] == case['orange'] and \
                        viable['blocking'] == case['blocking']:
                    number_same_cases += 1
                    case_distance = np.linalg.norm(viable['vector'] - case['vector'])
                if case_distance <= smallest_distance:
                    smallest_distance = case_distance
            if number_same_cases <= (max_cases - 1) and smallest_distance >= minimum_distance:
                print("adding case g", case['green'], 'o', case['orange'], 'b', case['blocking'], 'min distance',
                      smallest_distance)
                viables.append(case)
                continue
            if number_same_cases == 0:
                print("adding case g", case['green'], 'o', case['orange'], 'b', case['blocking'], 'min disancce',
                      smallest_distance)
                viables.append(case)
                continue

        minimum_distance -= 1

    chunks = []
    for case in viables:
        ck = ['isa', 'decision', 'green', int(case['green']),
              'orange', int(case['orange']), 'blocking', int(case['blocking']),
              'action', case['chosen_action']]
        # 'value_estimate',repr(case['value_estimate']),
        # 'vector',repr(list(case['fc1']))]

        chunks.append(ck.copy())
    pickle.dump(chunks, open(output_filepath, "wb"))



print("Done.")



