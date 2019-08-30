import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.models import Model, load_model
from keras.layers import LSTM, Input, Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import csv
import argparse
import keras.losses
import keras.metrics
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from keras.backend.tensorflow_backend import set_session, clear_session, get_session
from math import fsum 

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    print("Keras backend has been reset")
    
reset_keras()

data_path = "Models_Ange"

parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path

lstm_out = 82 # nombre de noeuds dans la "output layer"
batchSize = 50 # taille des lots de données
look_back = 197 # nombre de noeuds dans la "hidden layer"frg
inputsize = 162 # nombre de noeuds dans la "input layer"
skills = 81 # nb des différentes compétences évaluées chez les élèves

def prepross (xs):
        result = []
        for x in xs :
                xt_zeros = [0 for i in range(0, skills *2)]
                skill = np.argmax(x[1:])
                a = x[-1]
                pos = skill * 2 + int(a)
                xt = xt_zeros[:]
                xt[pos] = 1
                result.append(xt)
        return np.array(result)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)):
        for j in range(len(dataset[i]) - look_back-1) :
            dataX.append(prepross(dataset[i,j:(j+look_back)]))
            dataY.append(dataset[i , j+1:(j+ look_back+1)])
    return np.array(dataX), np.array(dataY)

def accur(y_true, y_pred):
    temp = y_true[:,:,0:-1]  * y_pred
    rel_pred = K.sum(temp, axis=2)
    return K.mean(K.equal(K.round(rel_pred), y_true[:,:,-1]))

#keras.metrics.accur = accur
    
def loss_function(y_true, y_pred):
     obs = y_true[:,:,-1]
     temp = y_true[:,:,0:-1] * y_pred
     rel_pred = K.sum(temp, axis=2)
     
#     rep12 = y_true[:,:,-1] * y_true[:,:,12]
#     rep16 = y_true[:,:,-1] * y_true[:,:,16]
#     rep61 = y_true[:,:,-1] * y_true[:,:,61]
#     rep74 = y_true[:,:,-1] * y_true[:,:,74]
     rep77 = y_true[:,:,-1] * y_true[:,:,77]

     zero = tf.constant(0, dtype=tf.float32)
     
     mask12 = tf.not_equal(temp[:,:,12], zero)
     mask16 = tf.not_equal(temp[:,:,16], zero)
     mask61 = tf.not_equal(temp[:,:,61], zero)
     mask74 = tf.not_equal(temp[:,:,74], zero)
     mask77 = tf.not_equal(temp[:,:,77], zero)
    
     return  K.binary_crossentropy(rel_pred, obs)  + 20*tf.where(mask77, K.binary_crossentropy(rel_pred, obs), tf.zeros_like(rep77))

#keras.losses.loss_function = loss_function 


#df = pd.read_csv('rawData.csv', header=None)
#data = (df.values)[:,1:]
#data = np.array([np.array([y[1:-1].split(', ') for y in x ]) for x in data])
#
##new_data =[]
##for i in range(len(data)):
##    inds = [i for i in range(len(data[i]))]
##    shuffle(inds)
##    new_data.append(data[i,inds])
##data = np.array(new_data)
#
#print(data.shape)
#
#X_data, Y_data = create_dataset(data, look_back)
#np.save('X_data.npy',X_data)
#np.save('Y_data.npy',Y_data)

if args.run_opt == 1:
    
    X_data = np.load('X_data.npy')
    Y_data = np.load('Y_data.npy')
    
    X_train1, X_test, Y_train1, Y_test1 = train_test_split(X_data,Y_data, test_size = 0.10, random_state = 42)
    X_train, X_val, Y_train2, Y_val1 = train_test_split(X_train1,Y_train1, test_size = 0.20, random_state = 42)
    
    Y_train = Y_train2[:,:,1:]
    Y_test = Y_test1[:,:,1:]
    Y_val = Y_val1[:,:,1:]


    lstm_layer = LSTM(81, batch_input_shape=(batchSize, look_back, inputsize), return_sequences=True)
    comment_input = Input(shape=(look_back,inputsize,),dtype='float32')
    x = lstm_layer(comment_input)
    preds =  TimeDistributed(Dense(81, activation='sigmoid'))(x)
    model = Model(inputs=comment_input,outputs=preds)
    model.compile(loss= loss_function, optimizer='adam', metrics=[accur])
    print(model.summary())
    num_epochs = 2
    print(X_train.shape,Y_train.shape)
    print(X_val.shape,Y_val.shape)
    print(X_test.shape,Y_test.shape)

    history = model.fit(X_train, Y_train,  validation_data=(X_val, Y_val), epochs = num_epochs, batch_size=batchSize)
    model.save("final_model_DKT_mask_Ange.hdf5")
    
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    #Test global avec l'AUC (Area under de curve)
    import numpy as np
    from sklearn.metrics import roc_auc_score
    pred  = model.predict(X_test)
    temp = Y_test.astype(np.float)[:,:,:-1] * pred
    y_true = ((Y_test[:,:,-1]).astype(np.float)).ravel()
    y_pred = (np.sum(temp, axis=2)).ravel()

    print("AUC = ")
    print(roc_auc_score(y_true.ravel(), y_pred.ravel()))
    
    #Test de chaque compétence en utilisant l'AUC
    zero = 0
    mask4 = ((np.equal(temp[:,:,77], zero)).astype(int)).ravel()
    import numpy.ma as ma
    pred_77 = ma.array((temp[:,:,77]).ravel(), mask=mask4)
    reel_77 = ma.array(((Y_test[:,:,-1]).astype(np.float)).ravel(), mask=mask4)
    pred_77 = pred_77.compressed()
    reel_77 = reel_77.compressed()
    print(pred_77)
    print(reel_77)
    print("AUC_77 = ")
    print(roc_auc_score(reel_77.ravel(), pred_77.ravel()))
