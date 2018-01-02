import numpy as np
import pandas as pd

result = np.load('feats/ENS_towrite.npy')
result = np.asarray(result, dtype = np.int32)
result = np.reshape(result, (result.size,1))
res = pd.DataFrame(result)
res.columns = ['LABEL']
res.to_csv('feats/Final_Test_Pred.csv',index = False )

np.savetxt('feats/Final_Test_Predtxt.txt',result)

