#!/usr/bin/env python
# coding: utf-8

# In[204]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from IPython.display import Image  
from sklearn import tree
import pydotplus
import graphviz
from sklearn.externals.six import StringIO 
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from scipy import stats
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import export_graphviz


# In[205]:


#Loading the data
data = np.loadtxt('Q1.csv',delimiter=',')
X=[]
Y=[]
U=[]
W=[]
for i in range(1000):
    if data[i,2]==-1:
        X.append(data[i,0])
        Y.append(data[i,1])
    elif data[i,2]==1:
        U.append(data[i,0])
        W.append(data[i,1])
#Scatter Plot of given data
a=plt.scatter(X,Y,color='red', marker='o')
b=plt.scatter(U,W,color='black', marker='x')
plt.legend((a,b),('Class -1','Class 1'))
plt.title("Scatter Plot of Given Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.show()


# In[206]:


# ID3 classification Tree uses entropy for split at each node
model= DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=11,min_impurity_split=0.01)
X_train=data[100:1000,0:2]
Y_train=data[100:1000,2]
X_test=data[0:100,0:2]
Y_test=data[0:100,2]
# Training the model
model.fit(X_train,Y_train )
#Testing the model
Y_predict = model.predict(X_test)
A=sklearn.metrics.accuracy_score(Y_test, Y_predict)
#Confusion Matrix
Q=pd.DataFrame(
    confusion_matrix(Y_test, Y_predict),
    columns=['Class -1', 'Class 1'],
    index=['True -1', 'True 1']
)
print(Q)
print(A)


# In[207]:


#Displaying Decision Tree
dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
dot_data = StringIO()
export_graphviz(model,out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[208]:


#Plotting Decision Boundary on the Scatter plot of Given training Data
X=[]
Y=[]
U=[]
W=[]
for i in range(1000):
    if data[i,2]==-1:
        X.append(data[i,0])
        Y.append(data[i,1])
    elif data[i,2]==1:
        U.append(data[i,0])
        W.append(data[i,1])

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min, y_max,0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
a=plt.scatter(X,Y,color='red', marker='o')
b=plt.scatter(U,W,color='black', marker='x')
cs = plt.contourf(xx, yy, Z,alpha=0.4)
plt.legend((a,b),('Class -1','Class 1'))
plt.title("Scatter Plot with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[238]:


#Creating a bagging Decision Tree solution using 7 trees
# By setting bootstrap= True we randomly sample with replacement the data to obtain a training population
bag=BaggingClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=11), n_estimators=7, max_samples=1.0)


# In[239]:


#Training the model
bag.fit(X_train,Y_train )


# In[240]:


li=[]
y=bag.predict(X_test)
A=sklearn.metrics.accuracy_score(Y_test, y)
li.append(A)
P=pd.DataFrame(confusion_matrix(Y_test, y),
        columns=['Class -1', 'Class 1'],
        index=['True -1', 'True 1'])
print(P)
print(A)


# In[244]:


for i in range(7):
    bag[i].fit(X_train,Y_train )


# In[245]:


li=[]
y_predict=[]


# In[ ]:





# In[246]:


for i in range(7):
    y=bag[i].predict(X_test)
    y_predict.append(y)
    A=sklearn.metrics.accuracy_score(Y_test, y)
    li.append(A)
    P=pd.DataFrame(
        confusion_matrix(Y_test, y),
        columns=['Class -1', 'Class 1'],
       index=['True -1', 'True 1']
    )
    print(P)


# In[247]:


li


# In[248]:


final=[]
for i in range(100):
    Y= stats.mode([y_predict[0][i],y_predict[1][i],y_predict[2][i],y_predict[3][i],y_predict[4][i],y_predict[5][i],y_predict[6][i]])
    final.append(Y[0][0])
print(final)


# In[249]:


R=pd.DataFrame(
       confusion_matrix(Y_test, final),
       columns=['Class -1', 'Class 1'],
       index=['True -1', 'True 1']
   )
print(R)


# In[250]:


X=[]
Y=[]
U=[]
W=[]
for i in range(1000):
    if data[i,2]==-1:
        X.append(data[i,0])
        Y.append(data[i,1])
    elif data[i,2]==1:
        U.append(data[i,0])
        W.append(data[i,1])
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min, y_max,0.1))
Z = bag.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
B = model.predict(np.c_[xx.ravel(), yy.ravel()])
B = B.reshape(xx.shape)
a=plt.scatter(X,Y,color='red', marker='o')
b=plt.scatter(U,W,color='black', marker='x')
cs = plt.contourf(xx, yy, Z,alpha=0.3)
ap = plt.contourf(xx, yy, B,alpha=0.3)
plt.legend((a,b),('Class -1','Class 1'))
plt.title("Scatter Plot with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[253]:


model1= DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=11)
ada=AdaBoostClassifier(base_estimator=model, n_estimators=7,algorithm='SAMME')


# In[254]:


ada=ada.fit(X_train,Y_train)


# In[256]:


y_predict=ada.predict(X_test)
A=sklearn.metrics.accuracy_score(Y_test, y_predict)
P=pd.DataFrame(
        confusion_matrix(Y_test, y_predict),
        columns=['Class -1', 'Class 1'],
        index=['True -1', 'True 1']
)
print(P)
print(A)


# In[257]:


a=ada.estimator_weights_ 
b=ada.estimator_errors_
T=[0,1,2,3,4,5,6]
print(a)
print(b)
au=plt.plot(T,a,a,'bo',label='weights')
bu=plt.plot(T,b,b,'ro',label='errors')
#plt.legend((au,bu),('Weights','errors'))
plt.legend()
plt.title("Plot of estimator error and estimator weight")
plt.xlabel("Tree number")


# In[258]:


X=[]
Y=[]
U=[]
W=[]
for i in range(1000):
    if data[i,2]==-1:
        X.append(data[i,0])
        Y.append(data[i,1])
    elif data[i,2]==1:
        U.append(data[i,0])
        W.append(data[i,1])
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min, y_max,0.1))
Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

a=plt.scatter(X,Y,color='red', marker='o')
b=plt.scatter(U,W,color='black', marker='x')
cs = plt.contourf(xx, yy, Z,alpha=0.5)
plt.legend((a,b),('Class -1','Class 1'))
plt.title("Scatter Plot with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[259]:


X=[]
Y=[]
U=[]
W=[]
for i in range(1000):
    if data[i,2]==-1:
        X.append(data[i,0])
        Y.append(data[i,1])
    elif data[i,2]==1:
        U.append(data[i,0])
        W.append(data[i,1])
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min, y_max,0.1))
Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
B = model.predict(np.c_[xx.ravel(), yy.ravel()])
B = B.reshape(xx.shape)
a=plt.scatter(X,Y,color='red', marker='o')
b=plt.scatter(U,W,color='black', marker='x')
cs = plt.contourf(xx, yy, Z,alpha=0.4)
cy = plt.contourf(xx, yy, B,alpha=0.4)
plt.legend((a,b),('Class -1','Class 1'))
plt.title("Scatter Plot with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[ ]:





# In[ ]:




