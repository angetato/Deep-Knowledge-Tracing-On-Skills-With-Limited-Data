import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import keras
from keras import backend as K
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
#from livelossplot import PlotLossesKeras

import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.layers import GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


from keras.layers import Input, Dense,multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from random import randint
import tensorflow as tf
import datetime

import attention_expert as atte

from random import shuffle



""" wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')  """

lstm_out = 17
batchSize = 200
look_back = 46
inputsize = 32
skills = 16

def prepross (xs):
        result = []
        for x in xs :
                xt_zeros = [0 for i in range(0, skills *2)]
                skill = np.argmax(x)
                a = x[-1]
                pos = skill * 2 + int(x[-1])
                xt = xt_zeros[:]
                xt[pos] = 1
                result.append(xt)
        return np.array(result)
# convert an array of values into a dataset matrix
def create_dataset(dataset, choix, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)):
        for j in range(len(dataset[i]) - look_back-1) :
            if choix == True :
                    a = prepross(dataset[i,j:(j+look_back)])
            else :
                    a = dataset[i,j:(j+look_back)]
            dataX.append(a)
            dataY.append(dataset[i , j+1:(j+ look_back+1)])
    return np.array(dataX), np.array(dataY)


def loss_function(y_true, y_pred):
        #f = lambda x : tf.where(x[0]>0, tf.where(x[1]>= 0.5,1,0),  0)
        """ f = lambda x : abs(x) > 0
        g = lambda x : abs(x) >= 0.5
        rel_pred = tf.cond(y_true>0, lambda : tf.cond(y_pred>= 0.5,1,0), lambda : 0)  """
        rep4 = y_true[:,:,-1] * y_true[:,:,4]
        # rep5 = y_true[:,:,-1] * y_true[:,:,5]
        # rep6 = y_true[:,:,-1] * y_true[:,:,6]
        # rep7 = y_true[:,:,-1] * y_true[:,:,7]
        # rep10 = y_true[:,:,-1] * y_true[:,:,10]
        # rep11 = y_true[:,:,-1] * y_true[:,:,11]
        # rep14 = y_true[:,:,-1] * y_true[:,:,14]
        # rep15 = y_true[:,:,-1] * y_true[:,:,15]
        obs = y_true[:,:,-1]
        temp = y_true[:,:,0:-1] * y_pred
        rel_pred = K.sum(temp, axis=2)
        zero = tf.constant(0, dtype=tf.float32)
        mask4 = tf.not_equal(temp[:,:,4], zero)
        mask5 = tf.not_equal(temp[:,:,5], zero)
        mask6 = tf.not_equal(temp[:,:,6], zero)
        mask7 = tf.not_equal(temp[:,:,7], zero)
        mask10 = tf.not_equal(temp[:,:,10], zero)
        mask11 = tf.not_equal(temp[:,:,11], zero)
        mask14 = tf.not_equal(temp[:,:,14], zero)
        mask15 = tf.not_equal(temp[:,:,15], zero)


        # keras implementation does a mean on the last dimension (axis=-1) which
        # it assumes is a singleton dimension. But in our context that would
        # be wrong.
        #+ 10*K.binary_crossentropy(rep5,(temp[:,:,5])) + 10*( K.binary_crossentropy(rep4, (temp[:,:,4]))  + K.binary_crossentropy(rep6, (temp[:,:,6]))  + K.binary_crossentropy(rep7, (temp[:,:,7]))  + K.binary_crossentropy(rep10, (temp[:,:,10]))  + K.binary_crossentropy(rep11, (temp[:,:,11]))  + K.binary_crossentropy(rep14, (temp[:,:,14]))  + K.binary_crossentropy(rep15, (temp[:,:,15])) ) 
        return  K.binary_crossentropy(rel_pred, obs) + 10*(tf.where(mask4, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4)) + tf.where(mask5, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4)) + tf.where(mask6, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4))+ tf.where(mask7, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4))+tf.where(mask10, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4))+tf.where(mask11, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4))+tf.where(mask14, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4))+tf.where(mask15, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep4))  )  

def accur(y_true, y_pred):

        temp = y_true[:,:,0:-1]  * y_pred
        rel_pred = K.sum(temp, axis=2)
        return K.mean(K.equal(K.round(rel_pred), y_true[:,:,-1]))

df = pd.read_csv('bn_data.csv')
bn = (df.values)[:,1:]
bn = np.array([np.array([ y[1:-2].split(', ') for y in x ]) for x in bn])
print(bn.shape)
df = pd.read_csv('rawData_kn.csv')
data = (df.values)[:,1:]
data = np.array([np.array([  y[1:-1].split(', ') for y in x ]) for x in data])
print(data.shape)
new_data =[]
for i in range(len(data)):
        inds = [i for i in range(len(data[i]))]
        shuffle(inds)
        new_data.append(data[i,inds])
data = np.array(new_data)
print(data.shape) 


X_bn, Y_bn = create_dataset(bn, False, look_back )
X_data, Y_data = create_dataset(data, True, look_back )

X = np.concatenate((X_data,X_bn), axis=-1)

print("taille des données = {}".format(X.shape))

ind_list = [i for i in range(len(X))]
shuffle(ind_list)
X_new  = X[ind_list, :]
Y_new = Y_data[ind_list,]

X_train1, X_test1, Y_train1, Y_test = train_test_split(X_new,Y_new, test_size = 0.10, random_state = 42)
X_train2, X_val1, Y_train, Y_val = train_test_split(X_train1,Y_train1, test_size = 0.20, random_state = 42)

X_train = X_train2[:,:,0:inputsize]
X_expert_train = X_train2[:,:,inputsize:inputsize +skills]
X_test = X_test1[:,:,0:inputsize]
X_expert_test = X_test1[:,:,inputsize:inputsize +skills]
X_val = X_val1[:,:,0:inputsize]
X_expert_val = X_val1[:,:,inputsize:inputsize +skills]

print(X_train.shape,X_expert_train.shape,Y_train.shape)
print(X_val.shape,X_expert_val.shape,Y_val.shape)
print(X_test.shape,X_expert_test.shape,Y_test.shape)




# define model


lstm_layer = LSTM(lstm_out-1, batch_input_shape=(batchSize, look_back, inputsize), return_sequences=True)

comment_input = Input(shape=(look_back,inputsize,), dtype='float32')
x = lstm_layer(comment_input)
expert_input = Input(shape=(look_back,16,), dtype='float32')
#merged = atte.AttentionE(16)([x,expert_input])
preds =  TimeDistributed(Dense(16, activation='sigmoid'))(x)
    
model = Model(inputs=[comment_input,expert_input], 
        outputs=preds)
model.compile(loss= loss_function,
        optimizer='adam',
        metrics=[accur])
print(model.summary())



#callbacks = [ PlotLossesKeras()]

history = model.fit([X_train,X_expert_train], Y_train,  validation_data=([X_val,X_expert_val], Y_val),epochs = 20, batch_size=batchSize)


scores = model.evaluate([X_test,X_expert_test], Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])




# make predictions
testPredict = model.predict([X_test,X_expert_test])
temp = Y_test[:,:,0:-1].astype(np.float) * testPredict
print(classification_report(Y_test[:,1,-1].astype(np.float) * Y_test[:,1,4].astype(np.float), np.round(temp[:,1,4])))
print(classification_report(Y_test[:,0,-1].astype(np.float) * Y_test[:,0,7].astype(np.float), np.round(temp[:,0,7])))
print(Y_test[:,1,4])
print(Y_test[:,0,4])
print(Y_test.shape)

""" for (i,j) in zip (testPredict[0:1,0:10,:], Y_test[0:1,0:10,:]):
    print("{} - {}".format(i,j))  """

""" np.save('testPredict.npy',testPredict[0:1,:,:])
np.save('y_test.npy',Y_test[0:1,:,:]) """

""" df = pd.DataFrame(testPredict[1])
df.to_csv('testPredict4.csv') 
df = pd.DataFrame(Y_test[1])
df.to_csv('y_test4.csv') 

df = pd.DataFrame(testPredict[2])
df.to_csv('testPredict5.csv') 
df = pd.DataFrame(Y_test[2])
df.to_csv('y_test5.csv')   """
  

