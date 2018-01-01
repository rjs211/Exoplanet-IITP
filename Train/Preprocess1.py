import numpy as np
import csv
#import matplotlib 
#from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from scipy.signal import medfilt


data = np.load('feats/CSVData.npy')
labels = np.load('feats/label0.npy')





def modClipper(sam , red = 0.02 , dist = 4):   # function to perform clipping of the top 2 % of the dataset to eliminate big maximas and noise 
    
    seqLen = sam.shape[-1]
    mod = int(seqLen*red)
    indrange = range(2*dist + 1)
    indrange = np.asarray(indrange , dtype = np.int32)
    for i in sam:
    	indSort = np.argsort(i) [-1*mod:]
    	l1 = []
    	lind1 = []
    	for j in indSort:
    	    repl = 0
    	    cnt = 0
    	    window = indrange + (j-dist)
    	    window = [ p for p in window if p >= 0 and p<seqLen and p != j]
    	    winLen = len(window)
    	    repl = np.average(i[window])
    	    #print(repl)
    	    
    	    l1.append(repl)
    	    lind1.append(j)
    	np.put(i , lind1, l1)
    
    return sam


x = range(data.shape[-1])   
y = data[0]
n = len(y)
t = float(1/float(n))
k = np.arange(n)
frq = k
nn = int(n/2)
frq = frq[range(nn)]
Y = np.fft.fft(y)/n

Y = abs(Y[range(nn)])


lidet = []
for i in data:
    #print(cnt)
    #cnt+=1
    y = medfilt(i, 51)
    z = i-y
    lidet.append(z)


print('Detrend Done')    
lidet = np.asarray(lidet,dtype = np.float32)
liclip = modClipper(lidet)
print('modclip done')
liclipnorm = preprocessing.normalize(liclip,axis = 1)
print('normalize done')
liclipscale = preprocessing.scale(liclip,axis = 1)
print('scalinf done')
print('Shape of clipnorm' , liclipnorm.shape)
np.save('feats/Det_clip.npy', liclip)
np.save('feats/Det_clip_norm.npy', liclipnorm)
np.save('feats/Det_clip_scale.npy', liclipscale)
lifft = []
print('saving Phase1 done')
for z in liclipscale:
    Y = np.fft.fft(z) / n
    Y = abs(Y[range(nn)])
    lifft.append(Y)



lifft = np.asarray(lifft, dtype = np.float32)
print('fft done')
print('Shape of clipscalefft' , lifft.shape)

np.save('feats/Det_clip_scale_fft.npy', lifft)

lifft = []
for z in liclipnorm:
    Y = np.fft.fft(z) / n
    Y = abs(Y[range(nn)])
    lifft.append(Y)



lifft = np.asarray(lifft, dtype = np.float32)
print('fft done')
print('Shape of clipnormfft' , lifft.shape)

np.save('feats/Det_clip_norm_fft.npy', lifft)











