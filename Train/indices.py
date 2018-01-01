import numpy as np
import sklearn as sk
import pickle
from sklearn.utils import shuffle





data = np.load('feats/CSVData.npy')
label = np.load('feats/label0.npy')
nums = data.shape[0]
ind = [i for i in range(nums)]
nums=(nums-30)//3
nums = int(nums)
#label = label-1
#np.save('feats/label0.npy', label)  # changing it to mod zero.
'''
indpos = ind[:30]
indneg = ind[30:]
'''
print(nums)

indpos = []
indneg = []
for i in ind:
    if label[i] == 1:
        indpos.append(i)
    else:
        indneg.append(i)

print(len(indpos),len(indneg))

poslen = len(indpos) //3
neglen = len(indneg) //3


indpos = shuffle(indpos,random_state = 0)
indneg = shuffle(indneg,random_state = 0)


indpostrain = indpos[poslen:]
indposTest = indpos[:poslen]
indnegtrain = indneg[neglen: ]
indnegTest = indneg[:neglen]

li = []
li.append(indpostrain)
li.append(indposTest)
li.append(indnegtrain)
li.append(indnegTest)

print(len(indpostrain),len(indposTest),len(indnegtrain),len(indnegTest))

#for i in li:
#    print(str(i))

with open('feats/trainTestInd.pkl','wb') as f:
    pickle.dump(li,f,protocol = pickle.DEFAULT_PROTOCOL )
    
    

