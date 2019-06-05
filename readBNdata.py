import pandas as pd
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('bn_data.csv')
kb = (data.values)[:,1:]

data = pd.read_csv('rawData.csv',sep=";")
responses = data.values

ind = [0, 41, 26, 14, 21, 1, 34, 42, 15, 31, 13, 33, 36, 20, 16, 45, 29, 9, 5, 3, 35, 18, 6, 2, 10, 30, 44, 19, 27, 40, 23, 8, 24, 28, 37, 39, 46, 25, 22, 4, 7, 17, 11, 47, 32, 12, 43, 38]
responses = responses[:,ind]

names = ['MPP_FFD','MPP_FMD','MTT_FFD','MTT_FMD','AC_FMA','DA_FMA','AC_FFA','DA_FFA','MPP_CCF','MTT_CCF','AC_CCF','DA_CCF','MPP_A','MTT_A','AC_A','DA_A']
questions = ['MPP_FFD_1','MPP_FFD_2','MPP_FFD_3','MPP_FMD_1','MPP_FMD_2','MPP_FMD_3','MPP_CCF_1','MPP_CCF_2','MPP_CCF_3','MPP_A_1','MPP_A_2','MPP_A_3','MTT_FFD_1','MTT_FFD_2','MTT_FFD_3','MTT_FMD_1','MTT_FMD_2','MTT_FMD_3','MTT_CCF_1','MTT_CCF_2','MTT_CCF_3','MTT_A_1','MTT_A_2','MTT_A_3','AC_FMA_1','AC_FMA_2','AC_FMA_3','AC_FFA_1','AC_FFA_2','AC_FFA_3','AC_CCF_1','AC_CCF_2','AC_CCF_3','AC_A_1','AC_A_2','AC_A_3','DA_FMA_1','DA_FMA_2','DA_FMA_3','DA_FFA_1','DA_FFA_2','DA_FFA_3','DA_CCF_1','DA_CCF_2','DA_CCF_3','DA_A_1','DA_A_2','DA_A_3']

questions = np.array(questions)[ind]

def f (rep):
    if rep == 0 :
        return -1 
    return rep
kb_time = []
kb=[]
temp = []
for student in range(len(responses)):
    for i,rep in zip(questions, responses[student]):
        if names[0] in i :
            temp = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,rep]
        elif names[1] in i  :
            temp = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,rep]
        elif names[8] in i  :
            temp = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,rep]
        elif names[12] in i  :
            temp = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,rep]
        elif names[2] in i  :
            temp = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,rep]
        elif names[3] in i  :
            temp = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,rep]
        elif names[9] in i  :
            temp = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,rep]
        elif names[13] in i  :
            temp = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,rep]
        elif names[4] in i  :
            temp = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,rep]
        elif names[6] in i  :
            temp = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,rep]
        elif names[10] in i  :
            temp = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,rep]
        elif names[14] in i  :
            temp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,rep]
        elif names[5] in i  :
            temp = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,rep]
        elif names[7] in i  :
            temp = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,rep]
        elif names[11] in i :
            temp = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,rep]
        elif names[15] in i  :
            temp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,rep]
        kb.append(temp)
    kb_time.append(kb)
    kb=[]

df = pd.DataFrame(kb_time)
df.to_csv('rawData_kn.csv') 
