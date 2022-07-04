import random
import math
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
def ground_truth(x,w,b):
    return (x*w)+b
def training_data(w,b):
    train_data=[]
    for i in range(20):
        x=random.randint(0,100)
        y= ground_truth(x,w,b)
        y= y+ random.choice((-1, 1))*random.uniform(0,y*0.1)
        train_data.append([x,y])
    return train_data;
def linear_regression(train_data,w,b):
    errb=0
    errw=0
    mse=0
    for i in range(len(train_data)):
        y= w*train_data[i][0]+b
        errb=errb+(train_data[i][1]-y)
        mse=mse+(train_data[i][1]-y)**2
        errw= errw+math.floor((train_data[i][1]-y)*train_data[i][0])
    errb=errb/len(train_data)
    errw=errw/len(train_data)
    mse= mse/len(train_data)
    return [errb,errw,mse]
def train_scatters(train_data):
    x=[]
    y=[]
    for i in range(len(train_data)):
        x.append(train_data[i][0])
        y.append(train_data[i][1])
    return [x,y]
def main():
    W=2
    B=3
    x=[0,100]
    n=0.000001
    y=[ground_truth(0,W,B),ground_truth(100,W,B)]
    mainline, = plt.plot(x,y,color='green',label="Ground Line")
    w=random.random()
    b=random.random()
    lx=[0,100]
    ly=[ground_truth(0,w,b),ground_truth(100,w,b)]
    line, = plt.plot(lx,ly,color='yellow',label="Linear Regression Line")
    #fig, ax = plt.subplots()
    x, y = [],[]
    sc = plt.scatter(x,y,color='black')
    #pnts, = plt.scatter([0,0], [0,0], color='r')
    train_data=training_data(W,B)
    points= train_scatters(train_data)
    sc.set_offsets(np.c_[points[0], points[1]])
    for i in range(500):
        results= linear_regression(train_data,w,b)
        print(f"Epoch {'%5d' %(i+1)}: mean square error={'%7.4lf' % results[0]} with line y={'%6.4lf' % w}x+{'%6.4lf' % b}")
        w=w+n*results[1]
        b=b+n*results[0]
        ly=[ground_truth(0,w,b),ground_truth(100,w,b)]
        line.set_data(lx,ly)
        plt.draw()
        plt.pause(0.05)
main()
