import pickle
import os




input_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/data/','actr_history_vector_test.p')
no_vectors = pickle.load(open(input_filepath,"rb"))

input_filepath = os.path.join('/Users/paulsomers/StarcraftMAC/MyAgents/data/','actr_history_vector.p')
with_vectors = pickle.load(open(input_filepath,"rb"))


no_vectors_correct = []
with_vectors_correct = []

for case in no_vectors:
    state = case[0]
    action = case[2]
    # selected beacon is always the correct answer
    if action == 'select_beacon':
        no_vectors_correct.append(1)
    else:
        no_vectors_correct.append(0)

    # if state['green'] and not state['orange'] and not state['blocking']:
    #     if action == 'select_beacon':
    #         with_vectors_correct.append(1)
    #     else:
    #         with_vectors_correct.append(0)
    # elif state['orange'] and not state['green'] and not state['blocking']:
    #     if action == 'select_beacon':
    #         with_vectors_correct.append(1)
    #     else:
    #         with_vectors_correct.append(0)
    # elif state['green'] and state['orange'] and state['blocking']:
    #     if action == 'select_beacon':
    #         no_vectors_correct.append(1)
    #     else:
    #         no_vectors_correct.append(0)
    # elif state['green'] and state['orange'] and not state['blocking']:
    #     if action == 'select_beacon':
    #         no_vectors_correct.append(1)
    #     else:
    #         no_vectors_correct.append(0)

for case in with_vectors:
    state = case[0]
    action = case[2]
    #selected beacon is always the correct answer
    if action == 'select_beacon':
        with_vectors_correct.append(1)
    else:
        with_vectors_correct.append(0)
    # if state['green'] and not state['orange'] and not state['blocking']:
    #     if action == 'select_beacon':
    #         with_vectors_correct.append(1)
    #     else:
    #         with_vectors_correct.append(0)
    # elif state['orange'] and not state['green'] and not state['blocking']:
    #     if action == 'select_beacon':
    #         with_vectors_correct.append(1)
    #     else:
    #         with_vectors_correct.append(0)
    # elif state['green'] and state['orange'] and state['blocking']:
    #     if action == 'select_beacon':
    #         with_vectors_correct.append(1)
    #     else:
    #         with_vectors_correct.append(0)
    # elif state['green'] and state['orange'] and not state['blocking']:
    #     if action == 'select_beacon':
    #         no_vectors_correct.append(1)
    #     else:
    #         no_vectors_correct.append(0)



no_vectors_total = sum(no_vectors_correct)
no_vectors_percent = float(sum(no_vectors_correct)) / len(no_vectors)


with_vectors_total = sum(with_vectors_correct)
with_vectors_percent = float(sum(with_vectors_correct))  / len(with_vectors)


# for case in history:
#     state = case[0]
#     action = case[2]
#     if state['green'] and not state['orange'] and not state['blocking']:
#         if action == 'select_green':
#             correct.append(1)
#         else:
#             correct.append(0)
#     elif state['orange'] and not state['green'] and not state['blocking']:
#         if action == "select_orange":
#             correct.append(1)
#         else:
#             correct.append(0)
#     elif state['green'] and state['orange'] and not state['blocking']:
#         if action == 'select_orange':
#             correct.append(1)
#         else:
#             correct.append(0)


print("DONE")
