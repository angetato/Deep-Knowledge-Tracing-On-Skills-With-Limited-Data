import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import keras

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
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

import attention_utils as att
import attention_expert as atte

from random import shuffle



""" wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')  """


df = pd.read_csv("verbatims_text.csv")
X_text = (df.iloc[:,1]).values
df = pd.read_csv("verbatims_labels_5.csv")
Y = (df.iloc[:,1:]).values


df = pd.read_csv("vectorsSimilarity64_old.csv", header=None)
X_expert = (df.iloc[:,:]).values
print("X_expert {}".format(X_expert.shape))
print(X_expert[0])

max_features = 1500
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X_text)
X = tokenizer.texts_to_sequences(X_text)
X = pad_sequences(X)
print("X SHAPE {}".format(X.shape))

embed_dim = 128
lstm_out = 64
maxSeqLength = 50
batchSize = 24
numClasses = 6

def as_keras_metric(method):
    import functools
    from keras import backend as K
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
auc_roc = as_keras_metric(tf.metrics.auc)
recall = as_keras_metric(tf.metrics.recall)

# define model

""" model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(numClasses,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy', auc_roc])
print(model.summary()) """

MAX_SEQUENCE_LENGTH = X.shape[1]
embedding_layer = Embedding(max_features,
        embed_dim,
        input_length = X.shape[1])

lstm_layer = LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences= embedding_layer(comment_input)
x = lstm_layer(embedded_sequences)
x = Dropout(0.2)(x)
merged2 = att.Attention(MAX_SEQUENCE_LENGTH)(x)
merged = Dense(numClasses, activation='relu')(merged2)
expert_input = Input(shape=(5,), dtype='float32')
merged3 = atte.AttentionE(5)([merged,expert_input])
merged = keras.layers.concatenate([merged3,merged],axis=-1)
merged = Dropout(0.4)(merged)
#merged = BatchNormalization()(merged)
preds = Dense(numClasses, activation='sigmoid')(merged)
model = Model(inputs=[comment_input,expert_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', auc_roc])
print(model.summary())
X = np.concatenate((X,X_expert), axis=1)


ind_list = [i for i in range(len(X))]
shuffle(ind_list)
X_new  = X[ind_list, :]
Y_new = Y[ind_list,]

X_train1, X_test1, Y_train, Y_test = train_test_split(X_new,Y_new, test_size = 0.25, random_state = 42)
X_train = X_train1[:,0:300]
X_expert_train = X_train1[:,300:305]
X_test = X_test1[:,0:300]
X_expert_test = X_test1[:,300:305]

print(X_train.shape,X_expert_train.shape,Y_train.shape)
print(X_test.shape,X_expert_test.shape,Y_test.shape)

#callbacks = [ PlotLossesKeras()]

history = model.fit([X_train,X_expert_train], Y_train,  validation_data=([X_test,X_expert_test], Y_test),
          epochs = 20, batch_size=batchSize)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.subplot(3, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for auc
plt.subplot(3, 1, 2)
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(3, 1, 3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.tight_layout()
plt.show()

scores = model.evaluate([X_test,X_expert_test], Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

