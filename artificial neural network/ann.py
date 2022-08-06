import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt; plt.ion()
def loadData(file_name):
    data= np.loadtxt(file_name,delimiter=',',dtype=int)
    return data

def dataPreprocessing(data):
    classLabel=data[:,-1]
    classLabel=classLabel.T
    normalize_data= (np.delete(data,-1,axis=1))/16
    normalize_data= normalize_data.T
    return [normalize_data, classLabel]
def intialize_weights():
    w1=np.random.rand(10,64)-0.5
    b1= np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2 = np.random.rand(10,1)-0.5
    return w1,b1,w2,b2
def ReLU(Z):
    return np.maximum(0,Z)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagation(w1,b1,w2,b2,x):
    z1=w1.dot(x)+b1
    a1=sigmoid(z1)
    z2=w2.dot(a1)+b2
    a2=sigmoid(z2)
    return z1,a1,z2,a2
def labelMatrix(y):
    label_matrix=np.zeros((y.size,y.max()+1))
    label_matrix[np.arange(y.size),y]=1
    label_matrix= label_matrix.T
    return label_matrix
def derive_ReLU(z):
    return z>0
def derived(z):
    return sigmoid(z)*(1-sigmoid(z))
    
def back_propagataion(z1,a1,z2,a2,w2,x,y):
    m= y.size
    label_matrix= labelMatrix(y)
    dz2= a2-label_matrix
    dw2 = 1/m *(dz2.dot(a1.T))
    db2= 1/m *(np.sum(dz2,0))
    dz1= w2.T.dot(dz2)*derived(a1)
    dw1 = 1/m *(dz1.dot(x.T))
    db1= 1/m *(np.sum(dz1,0))
    return dw1,db1,dw2,db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,n):
    w1=w1-n*dw1
    b1=b1-n*db1
    w2=w2-n*dw2
    b2=b2-n*db2
    return w1,b1,w2,b2
def get_predictions(a2):
    return np.argmax(a2,0)
def get_accuracy(predictions,y):
    #print(predictions,y)
    return np.sum(predictions==y)/y.size
def mse(predictions,y):
    #print(predictions,y)
    return np.mean(np.power((predictions-y),2))   
def neural_network(data,label,iterations,n):
    w1,b1,w2,b2= intialize_weights()
    x=[]
    y=[]
    z=[]
    for i in range(iterations):
        for j in range(3):
            z1,a1,z2,a2= forward_propagation(w1,b1,w2,b2,data)
            dw1,db1,dw2,db2=back_propagataion(z1,a1,z2,a2,w2,data,label)
            w1,b1,w2,b2=update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,n)
        if i%10==0:
            print("Accuracy: ", get_accuracy(get_predictions(a2),label))
            print("MSE: ", mse(get_predictions(a2),label))
            x.append(i)
            y.append(mse(get_predictions(a2),label))
            z.append(get_accuracy(get_predictions(a2),label)*100)
    print(labelMatrix(label)[1])
    sc= plt.plot(x,y,color='black')
    #ac= plt.plot(x,z,color='green') 
    return w1,b1,w2,b2

def main(sys):    
    processedData=dataPreprocessing(loadData("optdigits.tra"))
    w1,b1,w2,b2= neural_network(processedData[0],processedData[1],1000,0.1)
    processedData2=dataPreprocessing(loadData("optdigits.tes"))
    b1=np.resize(b1,(10,1797))
    b2=np.resize(b2,(10,1797))
    z1,a1,z2,a2= forward_propagation(w1,b1,w2,b2,processedData2[0])
    print("Accuracy Test: ", get_accuracy(get_predictions(a2),processedData2[1]))
main(sys)
