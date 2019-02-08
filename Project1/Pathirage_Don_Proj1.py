#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get Dataset

df = pd.read_excel('proj1Dataset.xlsx')  #read into dataframe
df.head()

x = np.array(df['Weight'])        #Predictor array
t = np.array(df['Horsepower'])    #Target array

emptyidx = np.argwhere(np.isnan(t))  #Clean up data
t = t[~np.isnan(t)]
x = np.delete(x, emptyidx)

x = np.reshape(x, (len(x),1))
t = np.reshape(t,(len(t),1))

# Normalize
x2 = x/np.mean(x)
plt.scatter(x,t)


# In[2]:


# Create Design Matrix

X = np.ones((len(x),1))
X = np.mat(np.hstack((x,X)))
T = np.mat(t)

# Closed Form Solution
    
weights = np.dot(np.linalg.pinv(np.dot(X.T,X)),np.dot(X.T,T))
y = np.dot(X,weights)

    


# In[3]:


# Plot Closed Form Solution
print(weights)
plt.figure(1)
plt.scatter(x,t)
x3 = np.linspace(1500,5500,500)
plt.plot(x,y)
plt.title("Closed Form Solution")
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend(['weights: ' + str(weights)])


# In[4]:


#Gradient Descent Solution
k = np.mean(x)
# x2 = x/np.mean(x)
# t2 = t

# x2 = x2.reshape(400,1)
# t2 = t2.reshape(400,1)

X[:,0] = X[:,0]/k


# In[5]:


def guess(w,x):  #Prediction
    return w[0] + w[1]*x


# In[20]:


def cost(X,T,w):  #Cost
    J = np.sum(np.power(T-X*w,2))
    
    return J


# In[21]:


cost(X,T,np.random.randn(2,1))


# In[51]:


def gradient(t,X,w):  #Compute gradient
#     grad = np.array([0.0,0.0])
#     n = len(X)
    grad = 2 * w.T * X.T * X - 2*t.T*X
#     for i in range(n):
#         grad[0] += -1*(T[i] - guess(w,X[i]))
#         grad[1] += -1*(T[i] - guess(w,X[i]))*X[i]        
    return grad.T


# In[52]:


gradient(T,X,np.random.randn(2,1))


# In[66]:


def gradientDescent(X,T,rho,epochs):  #Find gradient descent solution
    weight = np.random.randn(2,1)
    e = []
#     print(weight)
    for i in range(epochs):
        grad = gradient(T,X,weight)
        ce = cost(X,T,weight)
        
#         print(ce)
        weight = weight-rho*grad
        e.append(ce)
        
    #plt.plot(range(epochs),e)
    return weight


# In[74]:


weight = gradientDescent(X,T,rho=1e-3,epochs=5000)
# print(weight[0],weight[1])

# x_mesh = np.linspace()

y2 = np.dot(X,weight)
weight[0] = weight[0]/k
print(weight)


# In[83]:


plt.figure(2)       #Plot gradient descent solution
plt.scatter(x,t)
x3 = np.linspace(1500,5500,500).reshape(500,1)
x3 = np.hstack((x3,np.ones([500,1])))

y3 = x3*weight

plt.plot(x3[:,0],y3,color = 'red')
plt.title("Gradient Descent Solution")
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend(['weights: ' + str(weights)])
plt.show()

