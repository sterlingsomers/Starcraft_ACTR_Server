import pickle
import numpy as np

historys = []
dict_dms = []
hist = 'history_x'
dict_dm = 'dict_dm_x'
for num in range(0,9):

    historys.append(pickle.load(open(hist + repr(num) + '.p', 'rb')))
    dict_dms.append(pickle.load(open(dict_dm + repr(num) + '.p', 'rb')))

print("finished reading")

#overall match
matched = 0
total = 0
green = 0
greens = 0
orange = 0
oranges = 0
blocking = 0
blockings = 0
non_blocking = 0
non_blockings = 0
green_and_orange = 0
green_and_oranges = 0
b_so = 0
g_so = 0
o_sg = 0
nb_sg = 0
b_sg = 0
g_sa = 0
o_sa = 0
nb_sa = 0
b_sa = 0
#history,dms are matching files
for history in historys:
    #scenario is a scenario in a history
    for scenario in history:
        total += 1
        if scenario['actr'] == scenario['chosen_action']:
            matched += 1
        if scenario['green'] and not scenario['orange'] and not scenario['blocking']:
            greens += 1
            if scenario['actr'] == scenario['chosen_action']:
                green += 1
            if scenario['actr'] == 'SELECT_ORANGE':
                g_so += 1
            if scenario['actr'] == 'SELECT_AROUND':
                g_sa += 1
        if not scenario['green'] and scenario['orange'] and not scenario['blocking']:
            oranges += 1
            if scenario['actr'] == scenario['chosen_action']:
                orange += 1
            if scenario['actr'] == 'SELECT_GREEN':
                o_sg += 1
            if scenario['actr'] == 'SELECT_AROUND':
                o_sa += 1
        if scenario['green'] and scenario['orange'] and not scenario['blocking']:
            green_and_oranges += 1
            non_blockings += 1
            if scenario['actr'] == scenario['chosen_action']:
                green_and_orange += 1
                non_blocking += 1
            if scenario['actr'] == 'SELECT_GREEN':
                nb_sg += 1
            if scenario['actr'] == 'SELECT_AROUND':
                nb_sa += 1
        if scenario['green'] and scenario['orange'] and scenario['blocking']:
            green_and_oranges += 1
            blockings += 1
            if scenario['actr'] == scenario['chosen_action']:
                blocking += 1
                green_and_orange += 1
            if scenario['actr'] == 'SELECT_GREEN':
                b_sg += 1
            if scenario['actr'] == 'SELECT_ORANGE':
                b_so += 1
            if scenario['actr'] == 'SELECT_AROUND':
                b_sa += 1




print("total:",matched/total)
print("green:",green/greens)
print("orangee:",orange/oranges)
print("blocking:",blocking/blockings)
print("non-blocking",non_blocking/non_blockings)
print("green_and_oranges",green_and_orange/green_and_oranges)
print("g_so",g_so/greens)
print("g_sa",g_sa/greens)
print("o_sg",o_sg/oranges)
print(o_sg)
print("o_sa",o_sa/oranges)
print("nb_sg",nb_sg/non_blockings)
print("nb_sa",nb_sa/non_blockings)
print("b_so",b_so/blockings)
print("b_sg",b_sg/blockings)
print('b_sa',b_sa/blockings)

#known categories
known = [('TRUE','FALSE','FALSE','SELECT-GREEN'),
         ('FALSE', 'TRUE', 'FALSE', 'SELECT-ORANGE'),
         ('TRUE', 'TRUE', 'FALSE', 'SELECT-ORANGE'),
         ('TRUE', 'TRUE', 'TRUE', 'SELECT-AROUND')]
unknown_categories = [('TRUE', 'FALSE', 'FALSE', 'SELECT-ORANGE'),
                      ('FALSE', 'TRUE', 'FALSE', 'SELECT-GREEN'),
                      ('TRUE', 'TRUE', 'FALSE', 'SELECT-GREEN'),
                      ('TRUE', 'TRUE', 'TRUE', 'SELECT-GREEN'),
                      ('TRUE', 'FALSE', 'FALSE', 'SELECT-AROUND'),
                      ('FALSE', 'TRUE', 'FALSE', 'SELECT-AROUND'),
                      ('TRUE', 'TRUE', 'FALSE', 'SELECT-AROUND')]

unknown_categories_dict = {('TRUE', 'TRUE', 'TRUE', 'SELECT-ORANGE'):[],
                        ('TRUE', 'FALSE', 'FALSE', 'SELECT-ORANGE'):[],
                        ('FALSE', 'TRUE', 'FALSE', 'SELECT-GREEN'):[],
                        ('TRUE', 'TRUE', 'FALSE', 'SELECT-GREEN'):[],
                        ('TRUE', 'TRUE', 'TRUE', 'SELECT-GREEN'):[],
                        ('TRUE', 'FALSE', 'FALSE', 'SELECT-AROUND'): [],
                        ('FALSE', 'TRUE', 'FALSE', 'SELECT-AROUND'): [],
                        ('TRUE', 'TRUE', 'FALSE', 'SELECT-AROUND'): []}
#categories

for dms in dict_dms:

    for dm in dms:
        if dm in known:
            pass
        else:
            unknown_categories_dict[dm].append(len(dms[dm]))
    #for key in unknown_categories_dict:
    #    print(key, unknown_categories_dict[key])
    #print('----')


for key in unknown_categories_dict:
    print(key, sum(unknown_categories_dict[key]),np.mean(unknown_categories_dict[key]),
          max(unknown_categories_dict[key]), min(unknown_categories_dict[key]))

print("analysis complete")
