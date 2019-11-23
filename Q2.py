# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:37:26 2019

@author: Basil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = np.loadtxt('Q2train.csv',delimiter=',')
data_test = np.loadtxt('Q2test.csv',delimiter=',')
plt.plot(data_train[:,1],data_train[:,2],'.',ms=8,linestyle='--')
plt.xlabel('latitude')
plt.ylabel('Longitude')
plt.grid()
plt.title('Plot of give data')
plt.show()
score=[]
#au.show()
state=[0,0,0,0,0,0];
state=np.array(state)
A=np.array([[1,2,2,0,0,0],
            [0,1,2,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,2,2],
            [0,0,0,0,1,2],
            [0,0,0,0,0,1]])
C=np.array([[1,0,0,0,0,0],
            [0,0,0,1,0,0]])
#Training Kalman Filter and predicting test data
PI=np.array([[0.00001,0,0,0,0,0],[0,0.00001,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0.00001,0,0],[0,0,0,0,0.00001,0],[0,0,0,0,0,0]])
Q=3*np.eye(6)
R=2*np.eye(2)
xnp=[]
li=[]
                
for i in range(100):
    xnp=np.dot(A,state)
    pnp=np.dot(np.dot(A,PI),A.T) +Q
    K=np.dot(np.dot(pnp,C.T),np.linalg.inv(np.dot(np.dot(C,pnp),C.T) + R))
    true=[data_train[i,1],data_train[i,0]*2,2,data_train[i,2],data_train[i,0]*2,2]
    y=np.dot(C,true)
    xp=xnp+np.dot(K,(y-np.dot(C,xnp)))
    li.append(xp) #Output Sequence x[n]
    P=pnp-np.dot(K,np.dot(C,pnp))
    state=xp
    PI=P
#Prediction
state2=[0,0,0,0,0,0]
l2=[]
for i in range(100):        
    xnp=np.dot(A,state2)        
    pnp=np.dot(np.dot(A,PI),A.T) +Q
    K=np.dot(np.dot(pnp,C.T),np.linalg.inv(np.dot(np.dot(C,pnp),C.T) + R))
    true=[data_test[i,1],data_test[i,0]*2,2,data_test[i,2],data_test[i,0]*2,2]
    y=np.dot(C,true)
    xp=xnp+np.dot(K,(y-np.dot(C,xnp)))
    l2.append(xp)
Ba=[]
for i in range(100):
    asd=[l2[i][0],l2[i][3]]
    Ba.append(asd)
Ba=np.array(Ba)
plt.plot(Ba[:,0],Ba[:,1])
plt.xlabel('latitude')
plt.ylabel('Longitude')
plt.grid()
plt.title('Plot of interpolated data')
plt.show()
PI=np.array([[0.00001,0,0,0,0,0],[0,0.00001,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0.00001,0,0],[0,0,0,0,0.00001,0],[0,0,0,0,0,0]])
print(PI)
l=3
m=3
for l in range(1,15):
    for m in range(1,15):
        #Kalman Filter Code
        Q=l*np.eye(6)
        R=m*np.eye(2)
        xnp=[]
        li=[]
                
        for i in range(100):
            xnp=np.dot(A,state)
            pnp=np.dot(np.dot(A,PI),A.T) +Q
            K=np.dot(np.dot(pnp,C.T),np.linalg.inv(np.dot(np.dot(C,pnp),C.T) + R))
            true=[data_train[i,1],data_train[i,0]*2,2,data_train[i,2],data_train[i,0]*2,2]
            y=np.dot(C,true)
            xp=xnp+np.dot(K,(y-np.dot(C,xnp)))
            li.append(xp) #Output Sequence x[n]
            P=pnp-np.dot(K,np.dot(C,pnp))
            state=xp
            PI=P
        state2=[0,0,0,0,0,0]
        l2=[]
        for i in range(100):
             xnp=np.dot(A,state2)
             pnp=np.dot(np.dot(A,PI),A.T) +Q
             K=np.dot(np.dot(pnp,C.T),np.linalg.inv(np.dot(np.dot(C,pnp),C.T) + R))
             true=[data_test[i,1],data_test[i,0]*2,2,data_test[i,2],data_test[i,0]*2,2]
             y=np.dot(C,true)
             xp=xnp+np.dot(K,(y-np.dot(C,xnp)))
             l2.append(xp)
        Cross_validation=0    
        Cross_validation_score=[]
        for i in range(100):
           true=[data_test[i,1],data_test[i,0]*2,2,data_test[i,2],data_test[i,0]*2,2]
           Cross_validation=Cross_validation +np.abs(((l2[i][0]-true[0])+(l2[i][3]-true[3]))**2)
           Cross_validation_score.append(Cross_validation)
        Cross_validation_score= np.mean(Cross_validation_score)
        score.append([l,m,Cross_validation_score])
score=np.array(score)
scr_min=np.argmin(score,axis=0)[2]
Optimal_K=score[scr_min,0]
Optimal_S=score[scr_min,1]
Optimal_value=score[scr_min,2]
print('Optimal_K :',Optimal_K)
print('Optimal_S :',Optimal_S)
print('Optimal_value :',Optimal_value)

#print("Optimal :" ,score[scr_min,:])        
xx = np.linspace(1,14,14)
yy = np.linspace(1,14,14)
X,Y=np.meshgrid(xx,yy)
Z=score[:,2].reshape(14,14)
plt.contour(X,Y,Z)
plt.title('Contour of cross-validation metric')
plt.xlabel('Values of K')
plt.ylabel('Values of S')
plt.show()
m=plt.plot(data_train[:,1],data_train[:,2],'x',color='red',label='training data')
n=plt.plot(data_test[:,1],data_test[:,2],'x',color='blue',label='Test data')
v=plt.plot(Ba[:,0],Ba[:,1],label='predicted data using kalman')
plt.legend()
plt.xlabel('latitude')
plt.ylabel('Longitude')
plt.grid()
plt.title('Plot of give data')
plt.show()
    
#for i in range (10):
    #for j in range(10):
    
 