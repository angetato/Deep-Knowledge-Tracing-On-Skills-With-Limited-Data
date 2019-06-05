import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" proba = np.load('testPredict.npy')
questions = np.load('y_test.npy') """
labels = ['MPP_FFD','MPP_FMD','MTT_FFD','MTT_FMD','AC_FMA','DA_FMA','AC_FFA','DA_FFA','MPP_CCF','MTT_CCF','AC_CCF','DA_CCF','MPP_A','MTT_A','AC_A','DA_A']

questions = pd.read_csv("y_test40.csv")
questions = questions.iloc[:,1:]
names = []
for question in questions.values:
    pos = np.argmax(question[0:-1])
    skill = labels[pos]
    rep = question[-1]
    names.append((skill,rep))

print((np.array(names)).shape)
# import the data directly into a pandas dataframe
nba = pd.read_csv("testPredict40.csv")
nba = nba.iloc[:,1:]
#print(nba)
# remove index title
#nba.index.name = names
ids = {}
keys = range(len(names))
values = names
for i in keys:
        ids[i] = values[i]
#ids = {0:names[0],1:names[1], 2:names[2],3:names[3],4:names[4],5:names[5],6:names[6],7:names[7],8:names[8],9:names[9],10:names[10],11:names[11],12:names[12],13:names[13],14:names[14],15:names[15],16:names[16],17:names[17],18:names[18],19:names[19]}
nba.rename(index=ids, inplace=True)
# normalize data columns
nba_norm = nba #(nba - nba.mean()) / (nba.max() - nba.min())
# relabel columns

nba_norm.columns = labels
# set appropriate font and dpi
sns.set(font_scale=0.6)
sns.set_style({"savefig.dpi": 10000})
# plot it out
ax = sns.heatmap(nba, cmap=plt.cm.Blues, annot=True, linewidths=.1)
# set the x-axis labels on the top
ax.xaxis.tick_top()
# rotate the x-axis labels
plt.xticks(rotation=90)
# get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
fig = ax.get_figure()
# specify dimensions and save
fig.set_size_inches(10, 10)
fig.savefig("nba40.png")