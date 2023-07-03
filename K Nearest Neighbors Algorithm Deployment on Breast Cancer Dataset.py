#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the needed libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib','inline')


# In[2]:


# Get the Data
from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[5]:


# The dataset is presented in a dictionary form
cancer.keys()


# In[7]:


print(cancer['DESCR'])


# In[8]:


cancer['feature_names']


# In[11]:


# Dataframe for features
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()


# In[12]:


cancer['target']


# In[18]:


df_target=pd.DataFrame(cancer['target'],columns=['Cancer'])


# In[20]:


df_feat.head()


# In[21]:


df_target.head()


# In[24]:


# Standardizing the variables
from sklearn.preprocessing import StandardScaler


# In[25]:


scaler = StandardScaler()


# In[26]:


scaler.fit(df_feat)


# In[28]:


scaled_features = scaler.transform(df_feat)


# In[29]:


df_feat_scaled = pd.DataFrame(scaled_features,columns=df_feat.columns)


# In[30]:


df_feat_scaled.head()


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaled_features,np.ravel(df_target),test_size=0.30,random_state=105)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[40]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[35]:


knn.fit(X_train,y_train)


# In[36]:


pred=knn.predict(X_test)


# In[37]:


# Predictions and Evaluations
from sklearn.metrics import classification_report,confusion_matrix


# In[38]:


print(confusion_matrix(y_test,pred))


# In[39]:


print(classification_report(y_test,pred))


# In[63]:


# Choosing the K-Value
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[65]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title("Error Rate vs. K value")
plt.xlabel('K')
plt.ylabel("Error Rate")
#plt.show()


# In[58]:


# Comparing the new result to K=1
# With K=1
from sklearn.metrics import accuracy_score

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


# In[59]:


# With K=21
# According to the graph, we find minimum error at K=21

knn = KNeighborsClassifier(n_neighbors=21)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print("WITH K = 1")
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print("Accuracy Score = ",accuracy_score(y_test,pred))


# In[60]:


# Accuracy Score of K=21 is greater than when K=1

