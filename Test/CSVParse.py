import csv
import numpy as np

'''
For converting CSV data to numpy arrays for easier usability

'''
li = []
lilabel = []

with open('feats/FinalTest.csv', newline='') as csvfile:  # file name for parsing
    reader = csv.DictReader(csvfile)
    fields = reader.fieldnames   #getting fieldnames for parsing
    flabel = fields[0]  #label field
    fields = fields[1:]  #other fields
    for row in reader: 
        li2 = []
        for fn in fields:
            li2.append(float(row[fn]))
        li2 = np.asarray(li2,dtype = np.float32 )
        li.append(li2)
        lilabel.append(2)    #dummy for confusion matrix


livalues = li


labelTest = np.asarray(lilabel,dtype = np.int32 )
csvdataTest = np.asarray(livalues,dtype = np.float32)

np.save('feats/labels.npy', labelTest)  #saving the labels in Folder
np.save('feats/CSVData.npy', csvdataTest)   # saving the data in folder
labelTest = labelTest-1
np.save('feats/label0.npy', labelTest)  #saving the labels in Folder





