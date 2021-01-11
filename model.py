#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import pickle


# In[2]:


dataset = pd.read_csv('data.csv')
dataset


# In[3]:


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[4]:


X=dataset.drop(['salary'],axis=1)
X


# In[5]:


y=dataset.drop(['experience','test_score','interview_score'],axis=1)
y


# In[6]:


#Fitting model with trainig data
regressor.fit(X, y)


# In[7]:


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# In[8]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[10, 5,10]]))


# In[ ]:




