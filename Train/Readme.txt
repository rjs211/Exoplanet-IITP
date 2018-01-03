Execution Order for training from scratch : 
csvParse.py   # parse the csv file to npy format and label changed to 0 index
indices.py    #  split data as positive negative validation and testing
Preprocess1.py  # detrend normed scaled, Fourier transfomed

#the already preprocessed features used for best models can be Found in feats

########Training Starts here

svm2.py  #  Linear svm is trained
TimeFreq2.py  # ANN model is trained

#   after these steps,  save the model u want as the best model in the 'best' directory in the 
#   corresponding model Directory
#
#

loadcheck1.py   #  the trained and saved model is used to 
ensemble.py   # to combine svm and ANN

# the saved ensemble models for each iteration can be found in Model/ENS/
