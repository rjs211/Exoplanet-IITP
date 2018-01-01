import numpy as np
import sklearn as sk
import sklearn
from sklearn.utils import shuffle


def getxy(data , label , indpos,indneg,posTimes = 1 , if_shuffle = True ):
    posdata = data[indpos]
    poslabel = label[indpos]
    posind = indpos
    fposdata = data[indpos]
    fposlabel = label[indpos]
    fposind = indpos
    for i in range(posTimes -1) :
        fposdata = np.concatenate( (fposdata,posdata),axis = 0)
        fposlabel = np.concatenate((fposlabel,poslabel),axis = 0)
        fposind = fposind + posind
    
    allx = data[indneg]
    ally = label[indneg]
    allind = indneg
    #print(type(fposdata))
    #print(fposdata.shape )
    #print( fposlabel.shape)
    #print(allx.shape)
    #print(ally.shape)
    
    allx = np.concatenate((allx,fposdata), axis = 0)
    ally = np.concatenate((ally,fposlabel), axis = 0)
    allind = allind + fposind
    
    if if_shuffle == True:
        allx,ally,allind = shuffle(allx,ally,allind ,random_state = 0)
    
    return allx,ally,allind
