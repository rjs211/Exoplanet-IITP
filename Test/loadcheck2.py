import pickle
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from getdata import getxy

import numpy as np
import pickle
import sklearn as sk
import getdata
from getdata import getxy
import sys


import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops import array_ops
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from tensorflow.contrib.rnn import TimeFreqLSTMCell as TFLCell
from tensorflow.contrib.rnn import LSTMCell,BasicRNNCell
from tensorflow import layers as lay



d1 = np.load('feats/Ens_SVM_Pred.npy')
d2 = np.load('feats/Ens_ANN_yhat.npy')
d3 = np.load('feats/Ens_ANN_yp.npy')
ns = d1.shape
d1 = np.reshape(d1,(ns[0],1))
d2 = np.reshape(d2,(ns[0],1))
d3 = np.reshape(d3,(ns[0],1))

data = np.concatenate((d1,d2,d3) ,axis = 1)
label = np.load('feats/Ens_Label.npy')

print(data.shape,label.shape)
nind = data.shape[0]

fxtest = data
fytest = label

seqlen = fxtest.shape[1]


modelname = 'Model/ENS/best/Ens_Model10'
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(modelname+'.meta')
    new_saver.restore(sess, modelname)
    gph = tf.get_default_graph()
    #print([n.name for n in gph.as_graph_def().node])
    
    y_p = gph.get_tensor_by_name("Ens_y_p:0")
    yh2 = gph.get_tensor_by_name("Ens_yh2:0")
    accuracy = gph.get_tensor_by_name("Ens_accuracy:0")
    batch_ph = gph.get_tensor_by_name("Ens_batch_ph:0")
    target_ph = gph.get_tensor_by_name("Ens_target_ph:0")
    #keep_prob_ph = gph.get_tensor_by_name("keep_prob_ph:0") 
    
    
    
    val_acc, y_pred , y_h = sess.run([accuracy, y_p,yh2], feed_dict = {
                                                                       batch_ph: fxtest,
                                                                       target_ph: fytest})
    
    
    
    y_true = fytest
    np.save('feats/Ens_Predict_yp.npy',y_pred)
    np.save('feats/Ens_Predict_yhat.npy',y_h)
    nz = np.count_nonzero(y_pred)
    print('ANNout: Pos:',nz,'Neg:', y_pred.shape[0]-nz )
    #print(y_h[:25],'\n pred\n', y_pred[:25])
    #print("train_loss : {:.3f}, train_acc: {:.3f} ".format(loss_train, acc_train))
    #print ("Precision", sk.metrics.precision_score(y_true, y_pred), end =" , ")
    #print ("Recall", sk.metrics.recall_score(y_true, y_pred), end =" , ")
    #fs = sk.metrics.f1_score(y_true, y_pred)
    #print ("f1_score", fs )
    #print ("confusion_matrix")
    print (sk.metrics.confusion_matrix(y_true, y_pred))
    y1 = y_pred +1
    y1 = np.asarray(y1,dtype = np.int32)
    np.save('feats/ENS_towrite.npy',y1)
    
    
    
'''

Using TensorFlow backend.
2018-01-03 00:29:59.987013: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-03 00:29:59.987064: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
(2000, 3) (2000,)
ANNout: Pos: 13 Neg: 1987
[[1987   13]
 [   0    0]]


'''
    
    
    
    
