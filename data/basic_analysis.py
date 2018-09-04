import pickle
import os




input_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/data/','actr_history.p')
history = pickle.load(open(input_filepath,"rb"))


correct = []

for case in history:
    state = case[0]
    action = case[2]
    if state['green'] and not state['orange'] and not state['blocking']:
        if action == 'select_green':
            correct.append(1)
        else:
            correct.append(0)
    elif state['orange'] and not state['green'] and not state['blocking']:
        if action == "select_orange":
            correct.append(1)
        else:
            correct.append(0)
    elif state['green'] and state['orange'] and not state['blocking']:
        if action == 'select_orange':
            correct.append(1)
        else:
            correct.append(0)


print("DONE")
