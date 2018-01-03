import sklearn as sk
import numpy as np
from sklearn.svm import LinearSVC
import pickle
from sklearn.utils import shuffle
from Trusc import get_truScore
from sklearn.metrics import precision_recall_fscore_support
from getdata import getxy
from imblearn.over_sampling import SMOTE 
from Trusc import get_truScore


data = np.load('feats/Det_clip_scale_fft.npy')
label = np.load('feats/label0.npy')
with open('feats/trainTestInd.pkl', 'rb') as handle:
    ind = pickle.load(handle)

fxtr , fytr, findtr = getxy(data, label, ind[0], ind[2], 3)   # oversampled 3 times
#fxtr , fytr, findtr = getxy(data, label, ind[0]+ind[1], ind[2]+ind[3], 2)

fxtest, fytest , findtest =  getxy(data, label, ind[1], ind[3], 1,if_shuffle = False)   # no need for overssampling Valdiation data

sm = SMOTE(random_state=42)   #aplying smote synthesis for prediction
fxtr, fytr = sm.fit_sample(fxtr, fytr)   #  smote samples

fnumpy = np.asarray(findtest, dtype = np.int32)  
np.save('feats/SVM_ind.npy', fnumpy) 

print(fxtr.shape,fxtest.shape)

for i in range(1):
	#print("degree is",str(i+3))     
	clf = LinearSVC( )   #applying linear svm  ie it hyperplane is going to be a equation of plae with n-1 variables.
	clf.fit(fxtr, fytr)  #fitting

	y_pred = clf.predict(fxtest)  #predictinng
	#np.save('SVM_Pred.npy', y_pred)
	pre,rec,fsc,_ = precision_recall_fscore_support(fytest, y_pred, average = 'binary')  #obtaining precision recall fscore
	#,average = 'binary'

	print(' precision : {} , recall = {} , fscore :   '.format(pre,rec,fsc) ,end = '')   
	print(str(fsc))
	print(sk.metrics.confusion_matrix(fytest, y_pred))   #printing confusin matrix
	print('True Score is : ',str(get_truScore(sk.metrics.confusion_matrix(fytest, y_pred))))   #printing true Score  # True score included later
	
	with open('Model/SVM/svmModel'+str(i+1)+'.pkl','wb') as f:  #saving model
		pickle.dump(clf,f,protocol = pickle.DEFAULT_PROTOCOL )




'''
(5238, 1598) (1320, 1598)
degree is 3
 precision : 0.8571428571428571 , recall = 0.5 , fscore :   0.631578947368
[[1307    1]
 [   6    6]]
True Score is :  0.499235474006


'''
