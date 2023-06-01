#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import fetch_openml


# In[20]:


#Loading Data from the internet
mnist = fetch_openml(name = 'mnist_784',parser = 'auto') #I gave parser='auto' to ignore the warning


# In[30]:


#Exploring the Dataset
mnist #I didn't get the data in the array form below


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[46]:


#Seggregation of data into two variables
X, y = mnist.data,mnist.target
#Can also be written as X,y = mnist['data'],mnist['target']


# In[90]:


random_index = 1000
random_digit = np.array(X.iloc[random_index])
some_random_digit = random_digit.reshape(28, 28) #Original Image pixel size(28-28)

plt.imshow(some_random_digit, cmap='binary', interpolation='nearest')
plt.axis('off')
plt.show()


# In[53]:


#Splitting Data into training and testing data
x_train,x_test = X[:6000],X[6000:7000] #training = 6000 rows
y_train,y_test = y[:6000],y[6000:7000] #testing = 1000 rows (just for our convenience)


# In[56]:


#Shuffling the data just to improve the efficiency (not Compulsory)(but a good way)
shuffle_index = np.random.permutation(6000) #x_train and y_train are taken as Pandas dataframes
x_train, y_train = x_train.values, y_train.values  # Convert to NumPy arrays
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# In[92]:


y_train = y_train.astype(np.int8) #digit predictor
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==0)
y_test_2 = (y_test==0) #Here we are checking if the digit on the integer is 7 or not


# In[61]:


print(np.array(y_test_2)) #observe the True values below


# In[101]:


from sklearn import svm


# In[113]:


clf = svm.SVC()


# In[114]:


clf.fit(x_train,y_train_2) # we have to use y_train_2 because we made a digit predictor


# In[115]:


y_pred = clf.predict([random_digit])


# In[116]:


y_pred #Change the random_digit values or the y_train_2 digit values for more input testings
#Here the digit on the image is 6 and the digit predictor is checking for a 7


# In[117]:


#Evaluation of the Model
#Cross Validation Method
from sklearn.model_selection import cross_val_score


# In[118]:


a=cross_val_score(clf,x_train,y_train_2,cv=3,scoring='accuracy') #cv=3 means 3-fold cross validation
#meaning it will split the training data into 3 folds and perform the training and evaluation process three times.


# In[123]:


a.mean()


# In[ ]:




