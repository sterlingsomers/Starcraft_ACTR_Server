import pickle

#open the pickled analysis data pickle it into form:
#chunks = [['isa', 'decision', 'green', 'True', 'orange', 'True', 'between', 'True', 'action', 'select-around'],
#                  ['isa', 'decision', 'green', 'False', 'orange', 'True', 'between', 'False', 'action', 'select-orange'],
#                  ['isa', 'decision', 'green', 'True', 'orange', 'False', 'between', 'False', 'action', 'select-green'],
#                  ['isa', 'decision', 'green', 'True', 'orange', 'True', 'between', 'False', 'action', 'select-orange']]
#      #add random vectors

chunks = []
history = pickle.load(open('analysis_data_100_random.p',"rb"))

for case in history:
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


    print(len(chunks))


