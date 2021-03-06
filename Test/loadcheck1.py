import numpy as np
import pickle
import sklearn as sk
import getdata
from getdata import getxy
from sklearn.metrics import precision_recall_fscore_support


import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops import array_ops
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from tensorflow.contrib.rnn import TimeFreqLSTMCell as TFLCell
from tensorflow.contrib.rnn import LSTMCell,BasicRNNCell
from tensorflow import layers as lay



#from tensorflow.python.ops import Conv1DLSTMCell




with open('Model/SVM/best/svmModel1.pkl','rb') as f:
    clf = pickle.load(f) 
svmData = np.load('feats/Det_clip_scale_fft.npy')



data = np.load('feats/Det_clip_scale.npy')
label = np.load('feats/label0.npy')
#print(label[:50])
#with open('feats/trainTestInd.pkl', 'rb') as handle:
#    ind = pickle.load(handle)

#fxtr , fytr, Findtr = getxy(data, label, ind[0], ind[2], 40)
fxtest = data
fytest = label
findtest =  [i for i in range(data.shape[0])]

svmxtest = svmData
svmytest = label
svmindtest =  findtest

print (fxtest.shape, svmxtest.shape)

#fxtr = np.reshape(fxtr, (fxtr.shape[0],fxtr.shape[1],1) )
fxtest = np.reshape(fxtest, (fxtest.shape[0],fxtest.shape[1],1) )
print (fxtest.shape, svmxtest.shape)

totest = np.asarray(svmindtest, dtype = np.int32 )
totest2 = np.asarray(findtest, dtype = np.int32 )

#if np.array_equal(totest,totest2) :
#    print('The ordering of svm and ANN are Equal')
np.save('feats/Ens_Label.npy',fytest)
#else:
#    print('///   FATAL FLAW. CHECK INDEX OF SVM == INDEX HERE.//////////////')
#    #seqlen = fxtr.shape[1]



svmy_pred = clf.predict(svmxtest)
np.save('feats/Ens_SVM_Pred.npy', svmy_pred)
nz = np.count_nonzero(svmy_pred)
print('SVMOUT: Pos:',nz,'Neg:', svmy_pred.shape[0]-nz )
#pre,rec,fsc,_ = precision_recall_fscore_support(svmytest, svmy_pred, average = 'binary')
#,average = 'binary'

#print(' precision : {} , recall = {} , fscore :   '.format(pre,rec,fsc) ,end = '')
#print(str(fsc))
print(sk.metrics.confusion_matrix(svmytest, svmy_pred))




seqlen = fxtest.shape[1]
keep_prob = 0.5



batch_size = 32

num_epochs = 6
delta = 0.5
sumfsc = 0.0
maxfs = 0.0


modelname = 'Model/RCNN/best/rnn_2cnn_fc56'
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(modelname+'.meta')
    new_saver.restore(sess, modelname)
    gph = tf.get_default_graph()
    #print([n.name for n in gph.as_graph_def().node])
    
    y_p = gph.get_tensor_by_name("y_p:0")
    yh2 = gph.get_tensor_by_name("yh2:0")
    accuracy = gph.get_tensor_by_name("accuracy:0")
    batch_ph = gph.get_tensor_by_name("batch_ph:0")
    target_ph = gph.get_tensor_by_name("target_ph:0")
    keep_prob_ph = gph.get_tensor_by_name("keep_prob_ph:0") 
    
    
    
    val_acc, y_pred , y_h = sess.run([accuracy, y_p,yh2], feed_dict = {
                                                                       batch_ph: fxtest,
                                                                       target_ph: fytest,
                                                                       keep_prob_ph :1.0})
    
    
    
    y_true = fytest
    np.save('feats/Ens_ANN_yp.npy',y_pred)
    np.save('feats/Ens_ANN_yhat.npy',y_h)
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


'''

Using TensorFlow backend.
2018-01-03 00:28:38.886260: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-03 00:28:38.886312: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
(2000, 3197) (2000, 1598)
(2000, 3197, 1) (2000, 1598)
SVMOUT: Pos: 6 Neg: 1994
[[1994    6]
 [   0    0]]
ANNout: Pos: 7 Neg: 1993
[[1993    7]
 [   0    0]]

'''













