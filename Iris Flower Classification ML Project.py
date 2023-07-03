#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Check the versions of libraries
# Python version
import sys
print("Python : {}".format(sys.version))
# scipy
import scipy
print("scipy : {}".format(scipy.__version__))
# numpy
import numpy
print("numpy : {}".format(numpy.__version__))
# maplotlib
import matplotlib
print("matplotlib : {}".format(matplotlib.__version__))
# pandas
import pandas
print("pandas : {}".format(pandas.__version__))
# scikit-learn
import sklearn
print("scikit-learn : {}".format(sklearn.__version__))


# In[32]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# In[33]:


# UCI Machine Learning Repository is used to load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal_length','sepal_width','petal_length','petal_width','class' ]
dataset=pandas.read_csv(url,names=names)


# In[34]:


#Number of rows and columns in the dataset
print(dataset.shape)


# In[35]:


#Print the dataset
print(dataset.head(30))


# In[36]:


#Summary of each attribute of the dataset
print(dataset.describe())


# In[37]:


# Number of instances that belong to each class
print(dataset.groupby('class').size())


# In[38]:


# Univariant Plot to understand about each attribute
# Box  and viscous plot
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[39]:


#Histogram
dataset.hist()
plt.show()


# In[40]:


# Multi variant plot to understand about the relationship between different attributes
# Scatter plot
scatter_matrix(dataset)
plt.show()


# In[50]:


# Create a validation dataset(Training Dataset)
#  Model Training
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)


# In[51]:


# Create a Test Harness
# 10-fold cross validation is used to test the accuracy
# 9- train, 1-test = 10
seed=6
scoring = "accuracy" #ratio of number of correctly predicted instances divided by the total number of instances in the data set * 100 - giving a percentage.


# In[52]:


# Spot Check Algorithms
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

# Evaluate each model in turn
results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,shuffle=True,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s : %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)
    


# In[54]:


# Support Vector Machine Algorithm
svn=SVC()
svn.fit(X_train,Y_train)


# In[75]:


# Model Evaluation
# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
accuracy_score(Y_test,predictions)*100


# In[76]:


confusion_matrix(Y_test,predictions)


# In[59]:


# A detailed classification report
print(classification_report(Y_test,predictions))


# In[61]:


# Precision defines the ratio of true positives to the sum of true positive and false positives.
# Recall defines the ratio of true positive to the sum of true positive and false negative.
# F1-score is the mean of precision and recall value.

# Testing the model
X_new = numpy.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[62]:


# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(X_new)


# In[63]:


# Logistic Regression Algorithm
lr=LogisticRegression()
lr.fit(X_train,Y_train)


# In[64]:


y_pred=lr.predict(X_test)


# In[65]:


y_pred


# In[69]:


accuracy_score(Y_test,y_pred)*100


# In[77]:


confusion_matrix(Y_test,y_pred)


# In[78]:


print(classification_report(Y_test,y_pred))


# In[70]:


# Linear Discriminant Analysis Algorithm
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)


# In[71]:


y_predicted=lda.predict(X_test)


# In[72]:


y_predicted


# In[73]:


accuracy_score(Y_test,y_predicted)*100


# In[74]:


confusion_matrix(Y_test,y_predicted)


# In[79]:


print(classification_report(Y_test,y_predicted))

