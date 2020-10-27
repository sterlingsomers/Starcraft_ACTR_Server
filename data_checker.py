import pickle
import pandas as pd
import matplotlib.pyplot as plt


filenames = ['data_1_20.pkl','data_21_40.pkl', 'data_41_60.pkl','data_61_80.pkl','data_81_100.pkl',
             'data_101_120.pkl','data_121_140.pkl','data_141_160.pkl']#'data_1_10.pkl','data_11_20.pkl','data_21_30.pkl','data_31_40.pkl','data_41_50.pkl']
dataframes = []
for filename in filenames:
    dat = pickle.load(open(filename,'rb'))
    #original Data didn't have the correct answer. Score is harder to deal with.
    correct = []
    for i,orange in enumerate(dat['orange']):
        if orange and dat['target'][i] == 0:
            correct.append(1)
        elif dat['green'][i] and dat['target'][i] == 1 and not orange:
            correct.append(1)
        else:
            correct.append(0)
    dat['correct_decision'] = correct.copy()
    dataframes.append(pd.DataFrame(dat))


df = pd.concat(dataframes)

non_cog = df.loc[df['noise'] == 0.0]
cog = df.loc[df['noise'] >= 0.25]
non_grouped = non_cog.groupby('decision_index')['value'].agg({'Mean':'mean','Std':'std'})
cog_grouped = cog.groupby('decision_index')['value'].agg({'Mean':'mean','Std':'std'})
# grouped = df.groupby('decision_index').filter(lambda x: any(x['noise'] == 0.0))['correct_decision'].agg({'Mean':'mean','Std':'std'})
non_grouped.reset_index(inplace=True)
cog_grouped.reset_index(inplace=True)
plt.plot(non_grouped.index, non_grouped['Mean'])
plt.fill_between(non_grouped.index, y1=non_grouped['Mean']-non_grouped['Std'],y2=non_grouped['Mean']+non_grouped['Std'],alpha=0.5)

plt.plot(cog_grouped.index, cog_grouped['Mean'], c='r')
plt.fill_between(cog_grouped.index, y1=cog_grouped['Mean']-cog_grouped['Std'],y2=cog_grouped['Mean']+cog_grouped['Std'],color='r',alpha=0.5)

# grouped_filter_noise_decay = df.groupby('decision_index').filter(lambda x : x['noise'] ==0)['correct_decision'].agg({'Mean':'mean','Std':'std'})
# grouped_filter_noise_decay.reset_index(inplace=True)
# plt.plot(grouped_filter_noise_decay.index, grouped_filter_noise_decay.Mean)



print('debug')
