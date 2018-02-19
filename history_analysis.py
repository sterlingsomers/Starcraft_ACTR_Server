import pickle


import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import svm



history = pickle.load(open("history.p","rb"))
xs = []
ys = []
for case in history:
    if case['green'] and case['orange'] and case['blocking']:
        if case['actr'] == case['chosen_action']:
            xs.append(1)
        else:
            xs.append(0)

print("done.")
sum = 0
num = 0
proportion_correct = 0
print(xs, len(xs))
for x in xs:
    num += 1
    sum += x
    print(sum/num)


