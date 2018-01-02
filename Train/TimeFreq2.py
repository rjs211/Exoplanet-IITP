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



#from tensorflow.python.ops import Conv1DLSTMCell





data = np.load('feats/Det_clip_scale.npy')
label = np.load('feats/label0.npy')

print(data.shape)
with open('feats/trainTestInd.pkl', 'rb') as handle:
    ind = pickle.load(handle)
fxtr , fytr, Findtr = getxy(data, label, ind[0], ind[2], 40)
fxtest, fytest , Findtest =  getxy(data, label, ind[1], ind[3], 1)





print (fxtr.shape, fxtest.shape)

fxtr = np.reshape(fxtr, (fxtr.shape[0],fxtr.shape[1],1) )
fxtest = np.reshape(fxtest, (fxtest.shape[0],fxtest.shape[1],1) )
print (fxtr.shape, fxtest.shape)




seqlen = fxtr.shape[1]
tf.reset_default_graph()

batch_ph = tf.placeholder(tf.float32, [None, seqlen, 1],name = 'batch_ph')
target_ph = tf.placeholder(tf.float32, [None], name = 'target_ph')
keep_prob_ph = tf.placeholder(tf.float32,name = 'keep_prob_ph')
# = tf.placeholder(tf.int32, [None])

hidden_size = 10
hidden_size2 = 20
print(batch_ph.get_shape())
output1 ,dum = rnn(BasicRNNCell(hidden_size) , inputs = batch_ph , dtype = tf.float32) #  tf.contrib.rnn.LSTMCell
#output2 ,_ = rnn(TFLCell(hidden_size2 ,feature_size = 10, frequency_skip = 100) , inputs = tf.transpose(output1 , perm=[1, 0, 2]) , dtype = tf.float32)

#output2 = tf.transpose(output2, perm=[1, 0, 2])
#rnnout = output2[:, -1, :]
print(output1.get_shape(),dum.get_shape())

outputConv = lay.conv1d(output1,filters = 4,kernel_size = 50,strides = 20, padding = 'same', activation = tf.nn.relu)

print(outputConv.get_shape())

outputConv2 = lay.conv1d(outputConv,filters = 2,kernel_size = 20,strides = 5, padding = 'valid', activation = tf.nn.relu)

print('Conv2:  ',outputConv2.get_shape())
outflat = tf.contrib.layers.flatten(outputConv2)

dense1 = tf.layers.dense(inputs=outflat, units=10)
keep_prob = 0.5
drop = tf.nn.dropout(dense1 , keep_prob_ph)

dense2 = tf.layers.dense(inputs=drop, units=1)
y_hat = tf.squeeze(dense2 , name = 'y_hat')
y_p = tf.round(tf.sigmoid(y_hat), name = 'y_p')
yh2 = tf.sigmoid(y_hat, name = 'yh2')

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(yh2), target_ph), tf.float32), name = 'accuracy')

loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.sigmoid(y_hat), targets=target_ph, pos_weight = 5))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)



# Accuracy metric


batch_size = 32

num_epochs = 60
delta = 0.5
sumfsc = 0.0
maxfs = 0.0

#Fxtr , Fytr, Findtr = getxy(data, label, ind[0], ind[2], 10)
#Fxtest, Fytest , Findtest =  getxy(data, label, ind[1], ind[3], 1)

saver = tf.train.Saver(max_to_keep=100)
modelname = 'Model/RCNN/ANN_model'
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
        for b in range(num_batches):
            x_batch = fxtr[ b*batch_size: (b+1)*batch_size ]
            y_batch = fytr[ b*batch_size: (b+1)*batch_size ]
            #x_batch = np.reshape(x_batch,(x_batch.shape[0],x_batch.shape[1],1) )
            loss_tr , acc , _ = sess.run([loss,accuracy,optimizer], feed_dict = {
                                                                                 batch_ph: x_batch,
                                                                                 target_ph: y_batch,
                                                                                 keep_prob_ph : keep_prob})
            acc_train +=acc
            loss_train = loss_tr * delta + loss_train * (1 - delta)
        acc_train /= num_batches
        
        
        #print('Testing:')
        
        val_acc, y_pred , y_h = sess.run([accuracy, y_p,yh2], feed_dict = {
                                                                             batch_ph: fxtest,
                                                                             target_ph: fytest,
                                                                             keep_prob_ph :1.0})
        y_true = fytest
        #print(y_h[:25],'\n pred\n', y_pred[:25])
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
        print('True Score is : ',str(get_truScore(sk.metrics.confusion_matrix(y_true, y_pred))))  #just added
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
the output format may be sligtly modified as the model was saved very  long ago and further changes were made to this script'''


'''


2017-12-28 12:02:28.872612: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-12-28 12:02:28.872695: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
(3420, 3197) (1320, 3197)
(3420, 3197, 1) (1320, 3197, 1)
(?, 3197, 1)
(?, 3197, 10) (?, 10)
(?, 160, 4)
Conv2:   (?, 29, 2)
Start Learning...
epoch : 0	train_loss : 1.147, train_acc: 0.847 
Precision 0.15 , Recall 0.5 , f1_score 0.230769230769
confusion_matrix
[[1274   34]
 [   6    6]]
epoch : 1	train_loss : 1.057, train_acc: 0.968 
Precision 0.118644067797 , Recall 0.583333333333 , f1_score 0.197183098592
confusion_matrix
[[1256   52]
 [   5    7]]
epoch : 2	train_loss : 1.004, train_acc: 0.986 
Precision 0.545454545455 , Recall 0.5 , f1_score 0.521739130435
confusion_matrix
[[1303    5]
 [   6    6]]
epoch : 3	train_loss : 1.017, train_acc: 0.994 
Precision 0.4 , Recall 0.5 , f1_score 0.444444444444
confusion_matrix
[[1299    9]
 [   6    6]]
epoch : 4	train_loss : 1.002, train_acc: 0.994 
Precision 0.625 , Recall 0.416666666667 , f1_score 0.5
confusion_matrix
[[1305    3]
 [   7    5]]
epoch : 5	train_loss : 0.988, train_acc: 0.996 
Precision 0.5 , Recall 0.416666666667 , f1_score 0.454545454545
confusion_matrix
[[1303    5]
 [   7    5]]
epoch : 6	train_loss : 0.997, train_acc: 0.998 
Precision 0.5 , Recall 0.416666666667 , f1_score 0.454545454545
confusion_matrix
[[1303    5]
 [   7    5]]
epoch : 7	train_loss : 0.991, train_acc: 0.994 
Precision 0.714285714286 , Recall 0.416666666667 , f1_score 0.526315789474
confusion_matrix
[[1306    2]
 [   7    5]]
epoch : 8	train_loss : 0.989, train_acc: 0.997 
Precision 0.555555555556 , Recall 0.416666666667 , f1_score 0.47619047619
confusion_matrix
[[1304    4]
 [   7    5]]
epoch : 9	train_loss : 0.985, train_acc: 0.998 
Precision 0.5 , Recall 0.416666666667 , f1_score 0.454545454545
confusion_matrix
[[1303    5]
 [   7    5]]
epoch : 10	train_loss : 0.989, train_acc: 0.997 
Precision 0.714285714286 , Recall 0.416666666667 , f1_score 0.526315789474
confusion_matrix
[[1306    2]
 [   7    5]]
epoch : 11	train_loss : 0.989, train_acc: 0.997 
Precision 0.571428571429 , Recall 0.333333333333 , f1_score 0.421052631579
confusion_matrix
[[1305    3]
 [   8    4]]
epoch : 12	train_loss : 0.986, train_acc: 0.998 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 13	train_loss : 0.985, train_acc: 0.998 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 14	train_loss : 0.984, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 15	train_loss : 0.989, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 16	train_loss : 0.986, train_acc: 0.999 
Precision 1.0 , Recall 0.333333333333 , f1_score 0.5
confusion_matrix
[[1308    0]
 [   8    4]]
epoch : 17	train_loss : 0.988, train_acc: 0.998 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 18	train_loss : 0.984, train_acc: 0.998 
Precision 0.5 , Recall 0.416666666667 , f1_score 0.454545454545
confusion_matrix
[[1303    5]
 [   7    5]]
epoch : 19	train_loss : 0.984, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 20	train_loss : 0.988, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 21	train_loss : 0.986, train_acc: 1.000 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 22	train_loss : 0.984, train_acc: 0.999 
Precision 1.0 , Recall 0.333333333333 , f1_score 0.5
confusion_matrix
[[1308    0]
 [   8    4]]
epoch : 23	train_loss : 0.984, train_acc: 0.998 
Precision 0.666666666667 , Recall 0.5 , f1_score 0.571428571429
confusion_matrix
[[1305    3]
 [   6    6]]
epoch : 24	train_loss : 0.986, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.5 , f1_score 0.571428571429
confusion_matrix
[[1305    3]
 [   6    6]]
epoch : 25	train_loss : 0.985, train_acc: 0.999 
Precision 0.444444444444 , Recall 0.333333333333 , f1_score 0.380952380952
confusion_matrix
[[1303    5]
 [   8    4]]
epoch : 26	train_loss : 0.994, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 27	train_loss : 0.984, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 28	train_loss : 0.985, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 29	train_loss : 0.990, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 30	train_loss : 0.986, train_acc: 0.998 
Precision 0.555555555556 , Recall 0.416666666667 , f1_score 0.47619047619
confusion_matrix
[[1304    4]
 [   7    5]]
epoch : 31	train_loss : 0.984, train_acc: 0.999 
Precision 0.571428571429 , Recall 0.333333333333 , f1_score 0.421052631579
confusion_matrix
[[1305    3]
 [   8    4]]
epoch : 32	train_loss : 0.984, train_acc: 0.999 
Precision 0.571428571429 , Recall 0.333333333333 , f1_score 0.421052631579
confusion_matrix
[[1305    3]
 [   8    4]]
epoch : 33	train_loss : 0.985, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 34	train_loss : 0.985, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 35	train_loss : 0.984, train_acc: 1.000 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 36	train_loss : 0.984, train_acc: 0.999 
Precision 0.625 , Recall 0.416666666667 , f1_score 0.5
confusion_matrix
[[1305    3]
 [   7    5]]
epoch : 37	train_loss : 0.985, train_acc: 0.999 
Precision 1.0 , Recall 0.333333333333 , f1_score 0.5
confusion_matrix
[[1308    0]
 [   8    4]]
epoch : 38	train_loss : 0.984, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 39	train_loss : 0.990, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 40	train_loss : 0.985, train_acc: 0.999 
Precision 0.666666666667 , Recall 0.333333333333 , f1_score 0.444444444444
confusion_matrix
[[1306    2]
 [   8    4]]
epoch : 41	train_loss : 0.989, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 42	train_loss : 0.984, train_acc: 0.999 
Precision 1.0 , Recall 0.333333333333 , f1_score 0.5
confusion_matrix
[[1308    0]
 [   8    4]]
epoch : 43	train_loss : 0.987, train_acc: 0.999 
Precision 1.0 , Recall 0.333333333333 , f1_score 0.5
confusion_matrix
[[1308    0]
 [   8    4]]
epoch : 44	train_loss : 0.987, train_acc: 0.990 
Precision 0.444444444444 , Recall 0.666666666667 , f1_score 0.533333333333
confusion_matrix
[[1298   10]
 [   4    8]]
epoch : 45	train_loss : 0.985, train_acc: 0.996 
Precision 0.583333333333 , Recall 0.583333333333 , f1_score 0.583333333333
confusion_matrix
[[1303    5]
 [   5    7]]
epoch : 46	train_loss : 0.986, train_acc: 0.996 
Precision 0.5 , Recall 0.583333333333 , f1_score 0.538461538462
confusion_matrix
[[1301    7]
 [   5    7]]
epoch : 47	train_loss : 0.985, train_acc: 0.998 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 48	train_loss : 0.984, train_acc: 0.999 
Precision 0.583333333333 , Recall 0.583333333333 , f1_score 0.583333333333
confusion_matrix
[[1303    5]
 [   5    7]]
epoch : 49	train_loss : 1.000, train_acc: 0.999 
Precision 0.625 , Recall 0.416666666667 , f1_score 0.5
confusion_matrix
[[1305    3]
 [   7    5]]
epoch : 50	train_loss : 0.984, train_acc: 1.000 
Precision 0.833333333333 , Recall 0.416666666667 , f1_score 0.555555555556
confusion_matrix
[[1307    1]
 [   7    5]]
epoch : 51	train_loss : 0.984, train_acc: 0.999 
Precision 0.833333333333 , Recall 0.416666666667 , f1_score 0.555555555556
confusion_matrix
[[1307    1]
 [   7    5]]
epoch : 52	train_loss : 0.984, train_acc: 0.999 
Precision 0.833333333333 , Recall 0.416666666667 , f1_score 0.555555555556
confusion_matrix
[[1307    1]
 [   7    5]]
epoch : 53	train_loss : 0.984, train_acc: 1.000 
Precision 0.833333333333 , Recall 0.416666666667 , f1_score 0.555555555556
confusion_matrix
[[1307    1]
 [   7    5]]
epoch : 54	train_loss : 0.984, train_acc: 0.999 
Precision 1.0 , Recall 0.416666666667 , f1_score 0.588235294118
confusion_matrix
[[1308    0]
 [   7    5]]
epoch : 55	train_loss : 0.984, train_acc: 0.998 
Precision 1.0 , Recall 0.416666666667 , f1_score 0.588235294118
confusion_matrix
[[1308    0]
 [   7    5]]
epoch : 56	train_loss : 0.988, train_acc: 0.999 
Precision 1.0 , Recall 0.416666666667 , f1_score 0.588235294118
confusion_matrix
[[1308    0]
 [   7    5]]
epoch : 57	train_loss : 0.984, train_acc: 0.999 
Precision 1.0 , Recall 0.333333333333 , f1_score 0.5
confusion_matrix
[[1308    0]
 [   8    4]]
epoch : 58	train_loss : 0.985, train_acc: 0.999 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
epoch : 59	train_loss : 0.984, train_acc: 1.000 
Precision 0.8 , Recall 0.333333333333 , f1_score 0.470588235294
confusion_matrix
[[1307    1]
 [   8    4]]
Using TensorFlow backend.
Global Maximum is :   0.588235294118
avg fsc is :   0.480353176388

'''




