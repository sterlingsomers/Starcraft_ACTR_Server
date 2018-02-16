import pickle
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import svm


#open the pickled analysis data pickle it into form:
#chunks = [['isa', 'decision', 'green', 'True', 'orange', 'True', 'between', 'True', 'action', 'select-around'],
#                  ['isa', 'decision', 'green', 'False', 'orange', 'True', 'between', 'False', 'action', 'select-orange'],
#                  ['isa', 'decision', 'green', 'True', 'orange', 'False', 'between', 'False', 'action', 'select-green'],
#                  ['isa', 'decision', 'green', 'True', 'orange', 'True', 'between', 'False', 'action', 'select-orange']]
#      #add random vectors

chunks = []

history = pickle.load(open('analysis_data_100_random.p',"rb"))
print(type(history))
print(len(history))
current = None
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
                case_distance = np.linalg.norm(viable['fc1']-case['fc1'])
            if case_distance <= smallest_distance:
                smallest_distance = case_distance
        if number_same_cases <= (max_cases-1) and smallest_distance >= minimum_distance:
            print("adding case g",case['green'],'o',case['orange'],'b',case['blocking'],'min distance',smallest_distance)
            viables.append(case)
            continue
        if number_same_cases == 0:
            print("adding case g", case['green'], 'o', case['orange'], 'b', case['blocking'],'min disancce',smallest_distance)
            viables.append(case)
            continue

    minimum_distance -= 1

# while minimum_distance >= 0:
#     minimum_distance -= 1
#     for case in history:
#         if case['green'] and not case['orange'] and not case['blocking']:
#             if current == 'green_only':
#                 #try and avoid multiples of same scenario
#                 continue
#             if green_only >= 5:
#                 continue
#             print("green only")
#             green_only += 1
#             current = 'green_only'
#             for viable in viables:
#                 if viable['green'] and not viable['orange'] and not viable['blocking']:
#                     if np.linalg.norm(viable['fc1'] - case['fc1']) >= minimum_distance:
#                         #ed = np.linalg.norm(narray1 - narray2)
#                         print("adding green distance", np.linalg.norm(viable['fc1']-case['fc1']))
#                         viables.append(case)
#             if not viables:
#                 viables.append(case)
#             has_case = False
#             for viable in viables:
#                 if viable['green'] and not viable['orange'] and not viable['blocking']:
#                     has_case = True
#                     break
#             if not has_case:
#                 print("adding green only")
#                 viables.append(case)
#         elif case['green'] and case['orange'] and case['blocking']:
#             if current == 'green_orange_blocking':
#                 # try and avoid multiples of same scenario
#                 continue
#             if green_orange_blocking >= 5:
#                 continue
#             print("blocking")
#             green_orange_blocking += 1
#             current = 'green_orange_blocking'
#             for viable in viables:
#                 if viable['green'] and viable['orange'] and viable['blocking']:
#                     if np.linalg.norm(viable['fc1'] - case['fc1']) >= minimum_distance:
#                         #ed = np.linalg.norm(narray1 - narray2)
#                         print("adding blocking distance", np.linalg.norm(viable['fc1'] - case['fc1']))
#                         viables.append(case)
#             if not viables:
#                 viables.append(case)
#             has_case = False
#             for viable in viables:
#                 if viable['green'] and viable['orange'] and viable['blocking']:
#                     has_case = True
#                     break
#             if not has_case:
#                 print("adding blocking")
#                 viables.append(case)
#         elif case['green'] and case['orange'] and not case['blocking']:
#             if current == 'green_orange_non':
#                 # try and avoid multiples of same scenario
#                 continue
#             if green_orange_non >= 5:
#                 continue
#             print("non-blocking")
#             green_orange_non += 1
#             current = 'green_orange_non'
#             for viable in viables:
#                 if viable['green'] and viable['orange'] and not viable['blocking']:
#                     if np.linalg.norm(viable['fc1'] - case['fc1']) >= minimum_distance:
#                         # ed = np.linalg.norm(narray1 - narray2)
#                         print("adding not-blocking distance", np.linalg.norm(viable['fc1'] - case['fc1']))
#                         viables.append(case)
#             if not viables:
#                 viables.append(case)
#             has_case = False
#             for viable in viables:
#                 if viable['green'] and viable['orange'] and not viable['blocking']:
#                     has_case = True
#                     break
#             if not has_case:
#                 print("adding non-blocking")
#                 viables.append(case)
#         elif not case['green'] and case['orange'] and not case['blocking']:
#             if current == 'orange_only':
#                 # try and avoid multiples of same scenario
#                 continue
#             if orange_only >= 5:
#                 continue
#             print("orange_only")
#             orange_only += 1
#             current = 'orange_only'
#             for viable in viables:
#                 if not viable['green'] and viable['orange'] and not viable['blocking']:
#                     if np.linalg.norm(viable['fc1'] - case['fc1']) >= minimum_distance:
#                         # ed = np.linalg.norm(narray1 - narray2)
#                         print("adding orange distance", np.linalg.norm(viable['fc1'] - case['fc1']))
#                         viables.append(case)
#             if not viables:
#                 viables.append(case)
#             has_case = False
#             for viable in viables:
#                 if not viable['green'] and viable['orange'] and not viable['blocking']:
#                     has_case = True
#                     break
#             if not has_case:
#                 print("adding orange only")
#                 viables.append(case)

#print("g",green_only,"o",orange_only,"gob",green_orange_blocking,"gon",green_orange_non)

print("viables",len(viables))
for case in viables:
    ck = ['isa','decision','green',repr(case['green']),
          'orange',repr(case['orange']),'between',repr(case['blocking']),
          'value_estimate',repr(case['value_estimate']),
          'vector',repr(list(case['fc1']))]

    if case['green'] and case['orange'] and case['blocking']:
        ck.append('action')
        ck.append('select-around')
    else:
        ck.append('action')
        ck.append('select-beacon')
    chunks.append(ck.copy())


pickle.dump(chunks, open("chunks.p","wb"))

# range = [0]
#
# for case in history:
#     for sub_case in case['case_list']:
#         range.append(sub_case['euclidean'])
#
# [max,min] = [max]


