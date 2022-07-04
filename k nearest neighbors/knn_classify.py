import numpy as np
import math
import random
import sys
def loadData(file_name):
    data= np.loadtxt(file_name,dtype=int)
    return data
def dataPreprocessing(train_data,test_data):
    trainclassLabel=train_data[:,-1]
    testclassLabel=test_data[:,-1]
    normalize_train= (np.delete(train_data,-1,axis=1)-np.delete(train_data,-1,axis=1).mean(axis=0))/np.delete(train_data,-1,axis=1).std(axis=0)
    normalize_train= np.column_stack((normalize_train,trainclassLabel))
    normalize_test= (np.delete(test_data,-1,axis=1)-np.delete(train_data,-1,axis=1).mean(axis=0))/np.delete(train_data,-1,axis=1).std(axis=0)
    normalize_test= np.column_stack((normalize_test,testclassLabel))
    return [normalize_train,normalize_test]
def eucledianDistance(point1, point2):
    sum=0;
    for i in range (len(point1)-1):
        sum= sum + (point1[i]-point2[i])**2
    return math.sqrt(sum)

def getNeighbours(testpoint, traindata, index):
    mn=99999
    lb=-1
    idx=-1
    for i in range (len(traindata)):
        dist=eucledianDistance(testpoint,traindata[i])
        if(mn>dist and (i not in index)):
            mn=dist
            idx=i
            lb=traindata[i][len(traindata[i])-1]
    return [lb,idx];
def KNN(testdata,traindata,k):
    acc=0
    for i in range (len(testdata)):
        neighboursLabel=[]
        index=[]
        for j in range (k):
            neighbours= getNeighbours(testdata[i],traindata,index)
            neighboursLabel.append(int(neighbours[0]))
            index.append(neighbours[1])
        #label = max(neighboursLabel, key= neighboursLabel.count)
        #neighboursLabel=[1,1,1,3,3,3,5,6,6,6,9]
        y= np.bincount(np.array(neighboursLabel))
        maxLabel=max(y)
        kneigbours=[]
        for l in range(len(y)):
            if(y[l]==maxLabel):
                kneigbours.append(l)
        #print(kneigbours)     
        if(len(kneigbours)==1):
            label=kneigbours[0]
            #print("%3d" % (10));
            if(label==int(testdata[i][len(testdata[i])-1])):
                acc=acc+1
                print(f"ID={'%5d' % (i)}, predicted={'%3d' % (label)}, true={'%3d' % (int(testdata[i][len(testdata[i])-1]))}")
            else:
                print(f"ID={'%5d' % (i)}, predicted={'%3d' % (label)}, true={'%3d' % (int(testdata[i][len(testdata[i])-1]))}")
        else:
            label = kneigbours[random.randint(0, len(kneigbours)-1)]
            if(int(testdata[i][len(testdata[i])-1]) in kneigbours ):
                acr=1/len(kneigbours)
                acc=acc+acr
                print(f"ID={'%5d' % (i)}, predicted={'%3d' % (label)}, true={'%3d' % (int(testdata[i][len(testdata[i])-1]))}")
            else:
               print(f"ID={'%5d' % (i)}, predicted={'%3d' % (label)}, true={'%3d' % (int(testdata[i][len(testdata[i])-1]))}")
        #print(neighboursLabel)
    print(f"classification accuracy={'%6.4lf' % ((acc/len(testdata))*100)}")                 
        
def main(sys):    
    k= int(sys.argv[3])
    processedData=dataPreprocessing(loadData(sys.argv[1]), loadData(sys.argv[2]))
    testdata = processedData[1]
    traindata= processedData[0]
    KNN(testdata,traindata,k)
main(sys)
