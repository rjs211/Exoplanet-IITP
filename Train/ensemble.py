import pickle
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from getdata import getxy

import numpy as np
import pickle
import sklearn as sk
import getdata
from getdata import getxy
from Trusc import get_truScore
import sys


import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops import array_ops
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from tensorflow.contrib.rnn import TimeFreqLSTMCell as TFLCell
from tensorflow.contrib.rnn import LSTMCell,BasicRNNCell
from tensorflow import layers as lay



d1 = np.load('feats/Ens_SVM_Pred.npy')  # the binary svm predictions
d2 = np.load('feats/Ens_ANN_yhat.npy')   # the probability of a sample being true
d3 = np.load('feats/Ens_ANN_yp.npy')  # the binary ANN predition
ns = d1.shape
d1 = np.reshape(d1,(ns[0],1))   # reshape for Concatenation
d2 = np.reshape(d2,(ns[0],1))
d3 = np.reshape(d3,(ns[0],1))

data = np.concatenate((d1,d2,d3) ,axis = 1)  #Cocatenation
label = np.load('feats/Ens_Label.npy')  # labels

print(data.shape,label.shape)  


# Training and validation split
nind = data.shape[0]
ntest = nind//3

indf = [i for i in range(nind)]
indf = shuffle(indf, random_state = 1)
indtest = indf[:ntest]
indtrain = indf[ntest:]

indpos = [i for i in indtrain if label[i] == 1 ]
print(len(indtrain), len(indpos) )

fxtr , fytr, findtr = getxy(data, label, indpos, indtrain, 35)

fxtest = data[indtest]
fytest = label[indtest]



seqlen = fxtr.shape[1]                 # defining placeholders for ensemble model
batch_ph = tf.placeholder(tf.float32, [None,seqlen ],name = 'Ens_batch_ph')   
target_ph = tf.placeholder(tf.float32, [None], name = 'Ens_target_ph')
#keep_prob_ph = tf.placeholder(tf.float32,name = 'keep_prob_ph')

dense1 = tf.layers.dense(inputs=batch_ph, units=3)  #first hidden layer
dense2 = tf.layers.dense(inputs=dense1, units=2)    # seond hidden layer
denseout = tf.layers.dense(inputs=dense2, units=1)   #output layer

y_hat = tf.squeeze(denseout , name = 'Ens_y_hat')     
y_p = tf.round(tf.sigmoid(y_hat), name = 'Ens_y_p')
yh2 = tf.sigmoid(y_hat, name = 'Ens_yh2')

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(yh2), target_ph), tf.float32), name = 'Ens_accuracy')  

loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.sigmoid(y_hat), targets=target_ph, pos_weight = 5)) #weighted loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

batch_size = 32

num_epochs = 10 #10
delta = 0.5
sumfsc = 0.0
maxfs = 0.0

#Fxtr , Fytr, Findtr = getxy(data, label, ind[0], ind[2], 10)
#Fxtest, Fytest , Findtest =  getxy(data, label, ind[1], ind[3], 1)

saver = tf.train.Saver(max_to_keep=100)
modelname = 'Model/ENS/Ens_Model'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Start Learning...')
    
    for epoch in range(num_epochs):
        loss_train = 0
        loss_test = 0
        acc_train = 0
        acc_test = 0
        
        print("epoch : {}\t".format(epoch),end ="")
        
        num_batches = int(fxtr.shape[0] / batch_size)
        for b in range(num_batches):        #batch splitting
            x_batch = fxtr[ b*batch_size: (b+1)*batch_size ]
            y_batch = fytr[ b*batch_size: (b+1)*batch_size ]
            #x_batch = np.reshape(x_batch,(x_batch.shape[0],x_batch.shape[1],1) )
            loss_tr , acc , _ = sess.run([loss,accuracy,optimizer], feed_dict = {    
                                                                                 batch_ph: x_batch,
                                                                                 target_ph: y_batch})
            acc_train +=acc
            loss_train = loss_tr * delta + loss_train * (1 - delta)
        acc_train /= num_batches  #average accuracy
        
        
        #print('Testing:')
        
        val_acc, y_pred , y_h = sess.run([accuracy, y_p,yh2], feed_dict = {
                                                                             batch_ph: fxtest,
                                                                             target_ph: fytest})
        y_true = fytest
        #print(y_h[:25],'\n pred\n', y_pred[:25])
        #printing the metrics
        print("train_loss : {:.3f}, train_acc: {:.3f} ".format(loss_train, acc_train))
        print ("Precision", sk.metrics.precision_score(y_true, y_pred), end =" , ")
        print ("Recall", sk.metrics.recall_score(y_true, y_pred), end =" , ")
        fs = sk.metrics.f1_score(y_true, y_pred)
        print ("f1_score", fs )
        sumfsc += fs
        if fs > maxfs:
            maxfs = fs
        print ("confusion_matrix")
        print (sk.metrics.confusion_matrix(y_true, y_pred))
        print('True Score is : ',str(get_truScore(sk.metrics.confusion_matrix(y_true, y_pred))))
        sys.stdout.flush()
        nmodname = modelname+str(epoch+1)
        saver.save(sess, nmodname)
        
    #    new_tuples = []
        #for no in range(y_true.shape[0]):
        #    new_tuples.append( (Findtest[no] , y_pred[no], y_true[no]) )
    
   # testTuples.append(new_tuples)
    sumfsc /= float(num_epochs)  

print ("Global Maximum is :  " , maxfs)
print ("avg fsc is :  " , sumfsc)





'''

(3960, 3) (3960,)
2640 18
Start Learning...
epoch : 0	train_loss : 1.249, train_acc: 0.964 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 1	train_loss : 1.156, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 2	train_loss : 1.083, train_acc: 0.977 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 3	train_loss : 1.044, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 4	train_loss : 1.022, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 5	train_loss : 1.007, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 6	train_loss : 0.996, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 7	train_loss : 0.988, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 8	train_loss : 0.983, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 9	train_loss : 0.980, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
Global Maximum is :   0.965517241379
avg fsc is :   0.965517241379



Using TensorFlow backend.
2018-01-01 16:06:26.437169: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-01 16:06:26.437200: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
(3960, 3) (3960,)
2640 18
Start Learning...
epoch : 0	train_loss : 1.241, train_acc: 0.943 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
epoch : 1	train_loss : 1.139, train_acc: 0.967 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
epoch : 2	train_loss : 1.063, train_acc: 0.967 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
epoch : 3	train_loss : 1.025, train_acc: 0.967 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
epoch : 4	train_loss : 1.006, train_acc: 0.967 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
epoch : 5	train_loss : 0.995, train_acc: 0.972 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 6	train_loss : 0.988, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 7	train_loss : 0.983, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 8	train_loss : 0.980, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
epoch : 9	train_loss : 0.977, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
Global Maximum is :   0.965517241379
avg fsc is :   0.927203065134


'''

'''
Using TensorFlow backend.
2018-01-02 18:07:39.636403: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-02 18:07:39.636463: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
(3960, 3) (3960,)
2640 18
Start Learning...
epoch : 0	train_loss : 1.250, train_acc: 0.950 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
True Score is :  0.8
epoch : 1	train_loss : 1.155, train_acc: 0.967 
Precision 1.0 , Recall 0.8 , f1_score 0.888888888889
confusion_matrix
[[1305    0]
 [   3   12]]
True Score is :  0.8
epoch : 2	train_loss : 1.071, train_acc: 0.968 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 3	train_loss : 1.027, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 4	train_loss : 1.006, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 5	train_loss : 0.994, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 6	train_loss : 0.986, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 7	train_loss : 0.981, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 8	train_loss : 0.978, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
epoch : 9	train_loss : 0.976, train_acc: 0.978 
Precision 1.0 , Recall 0.933333333333 , f1_score 0.965517241379
confusion_matrix
[[1305    0]
 [   1   14]]
True Score is :  0.933333333333
Global Maximum is :   0.965517241379
avg fsc is :   0.950191570881

'''






