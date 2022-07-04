import random
import math
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
def ground_truth(x,w,b):
    return (x*w)+b
def training_data(w,b):
    train_data=[]
    for i in range(20):
        x=random.randint(0,1000)
        y=random.randint(0,1000)
        z=0;
        if ground_truth(x,w,b)<y:
            z=1
        else:
            z=-1
        train_data.append([x,y,z,0])
    return train_data
def train_scatters(train_data):
    x=[]
    y=[]
    for i in range(len(train_data)):
        x.append(train_data[i][0])
        y.append(train_data[i][1])
    return [x,y]
def perceptron_data(w,b, train_data):
    for i in range(len(train_data)):
        z=0
        if ground_truth(train_data[i][0],w,b)<train_data[i][1]:
            train_data[i][3]=1
        else:
            train_data[i][3]=-1
        
    return train_data
def missclassified(train_data):
    s=0
    for i in range (len(train_data)):
       if(train_data[i][2]!=train_data[i][3]):
           s=s+1
    return s
def main():
    W=2
    B=3
    x=[0,1000]
    n=0.0001
    y=[ground_truth(0,W,B),ground_truth(1000,W,B)]
    ax = plt.gca()
    ax.set_xlim([0, 1100])
    ax.set_ylim([0, 1100])
    mainline, = plt.plot(x,y,color='green',label="Ground Line")
    w=random.random()
    b=random.random()
    lx=[0,1000]
    ly=[ground_truth(0,w,b),ground_truth(1000,w,b)]
    line, = plt.plot(lx,ly,color='yellow',label="Perceptron Line")
    x, y = [],[]
    fx, fy=[],[]
    train_data=training_data(W,B)
    points= train_scatters(train_data)
    for i in range (len(train_data)):
        if(train_data[i][2]==-1):
            fx.append(train_data[i][0])
            fy.append(train_data[i][1])
        else:
            x.append(train_data[i][0])
            y.append(train_data[i][1])
            
    sc= plt.scatter(x,y,color='black')
    vc= plt.scatter(fx,fy,facecolors='none', edgecolors='black')
    train_data=perceptron_data(w,b,train_data)
    s=0
    while True:
        for j in range(len(train_data)):
            if(train_data[j][2]!=train_data[j][3]):
                d=0
                if ground_truth(train_data[j][0],w,b)<train_data[j][1]:
                    d=1
                else:
                    d=-1
                w=w+n*(d)*train_data[j][0]
                b=b+n*(d)*1
            ly=[ground_truth(0,w,b),ground_truth(1000,w,b)]
            line.set_data(lx,ly)
            plt.draw()
            plt.pause(0.05)
            #print(a,b,c)
        train_data=perceptron_data(w,b,train_data)
        m= missclassified(train_data)
        if(m==0):
            print(f"Epoch {'%5d' %(s+1)}: Missclassified Output {'%5d' %(m)}")
            break
        else:
            print(f"Epoch {'%5d' %(s+1)}: Missclassified Output {'%5d' %(m)}")
        s=s+1
            
main()
