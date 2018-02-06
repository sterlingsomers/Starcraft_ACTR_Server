import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import pickle


data = pickle.load(open("analysis_data.p", "rb"))
for x in range(0,len(data),3):
    plt.figure(x)
    cases = data[x]['case_list']
    xs = range(x)
    ys = []
   # if not x%20:
   #     c = input("c to continue, q to quit")
   #     if c == 'q':
   #         break
    
    plt.title("green:"+repr(data[x]['green'])+
              " orange:"+repr(data[x]['orange'])+
              " blocking:"+repr(data[x]['blocking']))
    for case in cases:
        ys.append(case['euclidean'])

    colors = []
    for case in cases:
        if case['green'] and case['orange'] and case['blocking']:
            colors.append('red')
        elif case['green'] and not case['orange'] and not case['blocking']:
            colors.append('green')
        elif case['green'] and case['orange']:
            colors.append('orange')
        else:
            colors.append('black')

    plt.scatter(xs,ys,c=colors)
plt.show()


