import pickle

a = pickle.load(open('data_test.pkl','rb'))

wrong = 0
for episode_i in range(len(a['chunk'])):
    if a['chunk'][episode_i]['value'] == -1:
        if 'wrong' in a['err'][episode_i]:
            wrong += 1
print('here')