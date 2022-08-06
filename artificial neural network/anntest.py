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

def intialize_weights(hidden_units):
    w1=np.random.rand(hidden_units,64)-0.5
    b1= np.random.rand(hidden_units,1)-0.5
    w2=np.random.rand(4,hidden_units)-0.5
    b2 = np.random.rand(4,1)-0.5
    return w1,b1,w2,b2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagation(w1,b1,w2,b2,x):   
    z1=w1.dot(x)+b1
    a1=sigmoid(z1)
    inpt=a1
    z2=w2.dot(inpt)+b2
    a2=sigmoid(z2)
    #inpt=a2
    return z1,a1,z2,a2

def labelMatrix(y):
    label_matrix=np.full((y.size,y.max()+1),0.1)
    label_matrix[np.arange(y.size),y]=0.9
    label_matrix= label_matrix.T
    #print(label_matrix.T.shape)
    return label_matrix

def derived_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
    
def back_propagataion(z1,a1,z2,a2,w2,x,y):
    m= y.size
    label_matrix= labelMatrix(y)
    dz2= a2-label_matrix
    dw2 = 1/m *(dz2.dot(a1.T))
    db2= 1/m *(np.sum(dz2,0))
    dz1= w2.T.dot(dz2)*derived_sigmoid(a1)
    dw1 = 1/m *(dz1.dot(x.T))
    db1= 1/m *(np.sum(dz1,0))
    return dw1,db1,dw2,db2

def update_weights(w1,b1,w2,b2,dw1,db1,dw2,db2,n):
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

def neural_network(data,label,epochs,hidden_unit,n):
    w1,b1,w2,b2= intialize_weights(hidden_unit)
    x=[]
    y=[]
    z=[]
    for i in range(epochs):
        for j in range(1):
            z1,a1,z2,a2= forward_propagation(w1,b1,w2,b2,data)
            dw1,db1,dw2,db2=back_propagataion(z1,a1,z2,a2,w2,data,label)
            w1,b1,w2,b2=update_weights(w1,b1,w2,b2,dw1,db1,dw2,db2,n)
        if i%10==0:
            mseror=mse(a2.T,labelMatrix(label).T)
            if(i==0):
                e=i+1
            else:
                e=i
            print(f"ID={'%5d' % (e)}, mean square error= {'%6.5lf' % (mseror)}")
            #print("Accuracy: ", get_accuracy(a2,labelMatrix(label)))
            #print("MSE: ", mse(a2,labelMatrix(label)))
            x.append(i)
            y.append(mse(a2,labelMatrix(label)))
            z.append(get_accuracy(get_predictions(a2),label)*100)
    print(f"ID={'%5d' % (i+1)}, mean square error= {'%6.5lf' % (mseror)}")
    x.append(i)
    y.append(mse(a2,labelMatrix(label)))
    #print(a2.T)
    #print(labelMatrix(label).T)
    sc= plt.plot(x,y,color='black')
    #ac= plt.plot(x,z,color='green') 
    return w1,b1,w2,b2

def main(sys):    
    processedData=dataPreprocessing(loadData(sys.argv[1]))
    w1,b1,w2,b2= neural_network(processedData[0],processedData[1],int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]))
    processedData2=dataPreprocessing(loadData(sys.argv[2]))
    m,n=processedData2[0].shape
    b1=np.resize(b1,(int(sys.argv[4]),n))
    b2=np.resize(b2,(4,n))
    z1,a1,z2,a2= forward_propagation(w1,b1,w2,b2,processedData2[0])
    print("Accuracy on Test Dataset: ", get_accuracy(get_predictions(a2),processedData2[1]))

main(sys)
