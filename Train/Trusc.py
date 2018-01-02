import sklearn as sk
import numpy as np
from sklearn.svm import LinearSVC
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from getdata import getxy



def get_truScore(cm) :
    
    num = (cm[0][0] * cm[1][1]) - (cm[1][0] * cm[0][1])
    den = (cm[1][0] + cm[1][1])*(cm[0][0] + cm[0][1])
    
    res = num/den
    
    return res
