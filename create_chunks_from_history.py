import pickle
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import svm
import os
import episode_utils


input_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/data/','history.p')
output_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/', 'chunks.p')

minimum_history_length = 10
maximum_history_length = 10

history = pickle.load(open(input_filepath,"rb"))

#don't need every single step. Want first step
history = episode_utils.filter_substeps(history)

if len(history) < minimum_history_length:
    raise ValueError("History does not meet minimum length requirements")
history = np.random.choice(history,maximum_history_length)

chunks = []
for case in history:
    ck = ['isa','decision','green',repr(case['green']),
          'orange',repr(case['orange']),'between',repr(case['blocking']),
          'action', case['chosen_action']]
          #'value_estimate',repr(case['value_estimate']),
          #'vector',repr(list(case['fc1']))]



    chunks.append(ck.copy())


pickle.dump(chunks, open(output_filepath,"wb"))


print("Done.")



