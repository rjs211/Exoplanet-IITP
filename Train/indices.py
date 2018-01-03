import numpy as np
import sklearn as sk
import pickle
from sklearn.utils import shuffle


#///////////////////  Test here dennotes validation set///////////


data = np.load('feats/CSVData.npy')   #input data
label = np.load('feats/label0.npy')   #input labels
nums = data.shape[0]
ind = [i for i in range(nums)]  #creating all indices
nums=(nums-30)//3
nums = int(nums) 
#label = label-1
#np.save('feats/label0.npy', label)  # changing it to mod zero.
'''
indpos = ind[:30]
indneg = ind[30:]
'''
print(nums)

indpos = []  #for storig positive indices
indneg = []  #for storing negative indices
for i in ind:       #separating based on label
    if label[i] == 1:   
        indpos.append(i)    
    else:
        indneg.append(i)

print(len(indpos),len(indneg))

poslen = len(indpos) //3       # Validation split
neglen = len(indneg) //3       # validation split


indpos = shuffle(indpos,random_state = 0)    #random shuffling
indneg = shuffle(indneg,random_state = 0)


indpostrain = indpos[poslen:]    # assigning the indices for each task
indposTest = indpos[:poslen]
indnegtrain = indneg[neglen:]
indnegTest = indneg[:neglen]

li = []    #list of list of index values for various labels and train and Validatoin  
li.append(indpostrain)
li.append(indposTest)
li.append(indnegtrain)
li.append(indnegTest)

print(len(indpostrain),len(indposTest),len(indnegtrain),len(indnegTest))

#for i in li:
#    print(str(i))

#Saving as a pickle for easy access.

with open('feats/trainTestInd.pkl','wb') as f:
    pickle.dump(li,f,protocol = pickle.DEFAULT_PROTOCOL )
    
    

