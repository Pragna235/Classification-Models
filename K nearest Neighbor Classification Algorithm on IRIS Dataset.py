#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib','inline')


# In[3]:


# IRIS Data Set

# The dataset consists of 50 samples from each of three species of Iris (Iris)

#The Iris Setose
from IPython.display import Image
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Iris_setosa.JPG/675px-Iris_setosa.JPG'
Image(url,width=300,height=300)


# In[4]:


# The Iris Versicolor
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/330px-Blue_Flag%2C_Ottawa.jpg'
Image(url,width=300,height=300)


# In[7]:


# The Iris Verginica
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/1104px-Iris_virginica.jpg'
Image(url,width=300,height=300)


# In[9]:


import seaborn as sns
iris = sns.load_dataset('iris')


# In[10]:


iris.head()


# In[11]:


# Exploratory data analysis

#Setosa is the most separable
sns.pairplot(iris,hue='species',palette='Dark2')


# In[28]:


# Create a kde plot of sepal_length versus width for setosa species of flower

import seaborn as sns

setosa = iris[iris['species'] == 'setosa']
sns.jointplot(x='sepal_width', y='sepal_length', data=setosa, kind='kde', cmap='plasma', fill=True, thresh=0.05)


# In[29]:


# Standardizing the variables

from sklearn.preprocessing import StandardScaler


# In[30]:


scaler = StandardScaler()


# In[31]:


scaler.fit(iris.drop("species",axis=1))


# In[32]:


scaled_features = scaler.transform(iris.drop('species',axis = 1))


# In[33]:


iris_feat = pd.DataFrame(scaled_features,columns = iris.columns[:-1])
iris_feat.head()


# In[35]:


# Train Test Split
   
from sklearn.model_selection import train_test_split


# In[37]:


X_train,X_test,y_train,y_test = train_test_split(scaled_features,iris['species'],test_size=0.30,random_state=103)


# In[38]:


# Using KNN Now
from sklearn.neighbors import KNeighborsClassifier


# In[39]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[40]:


knn.fit(X_train,y_train)


# In[41]:


pred = knn.predict(X_test)


# In[42]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[43]:


print(confusion_matrix(y_test,pred))


# In[44]:


print(classification_report(y_test,pred))



# In[45]:


# Choosing the K-Value
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[46]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title("Error Rate vs. K value")
plt.xlabel('K')
plt.ylabel("Error Rate")


# In[49]:


# With K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print("WITH K = 1")
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print("Accuracy Score = ",accuracy_score(y_test,pred))


# In[55]:


# With K=3 or 5 or 11
# According to the graph, we find minimum error at K=21

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print("WITH K = 1")
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print("Accuracy Score = ",accuracy_score(y_test,pred))

