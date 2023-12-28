#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Importing DATA

# In[2]:


data = pd.read_csv('weather.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# # Missing Data 
# ##### I have imputed the data using mode of the column

# In[5]:


data.isnull().any()


# In[6]:


data.isnull().sum()


# In[7]:


data.fillna(data.mode,inplace=True)


# In[8]:


data.isnull().sum()


# # Data visualization

# In[9]:


sns.pairplot(data[['MinTemp', 'MaxTemp', 'Rainfall']])


# In[10]:


sns.pairplot(data[['MinTemp', 'MaxTemp', 'Evaporation']])


# In[11]:


data.head()


# # From the data we can see 
# ### that we have data about the temparatures of many days and wind speed and rainfall on the day and rain tomorrow from this 
# 
# ### we can make a model that can predict how much rain (REGRESSION) or 
# 
# ### we can even predict if there will be rain tomorrow (CLASSIFICATION)

# # Linear Rigression

# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[13]:


X_train, X_test,y_train,y_test = train_test_split(data[['MinTemp','MaxTemp']],data['Rainfall'],test_size=0.3,random_state = 42)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


reg = LinearRegression()


# In[17]:


reg.fit(X_train,y_train)


# In[18]:


y_pred = reg.predict(X_test)


# In[19]:


mse = mean_squared_error(y_test,y_pred)


# In[20]:


print('Mean Squared error :',mse)


# In[21]:


y_pred


# ####  Above are the predictions of the amount of rain fall  a regression problem 

# # Logistic Regression 

# ### we try to predict if its gonna rain today even though there are many other features that  can decide rain fall I am just using two features out of them

# In[ ]:





# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import  confusion_matrix


# In[23]:


X = data[['MinTemp','MaxTemp']]


# In[24]:


ohe = OneHotEncoder(sparse_output= False, handle_unknown= 'ignore',drop= 'first')
y = ohe.fit_transform(data[['RainToday']])


# In[25]:


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)


# In[26]:


reg = LogisticRegression()


# In[27]:


reg.fit(X_train,y_train)


# In[28]:


y_pred = reg.predict(X_test)


# In[29]:


y_pred


# In[30]:


score = accuracy_score(y_test,y_pred)


# In[31]:


print('Accuracy :' ,score)


# In[32]:


confusion_matrix(y_test,y_pred)


# In[33]:


import pickle


# In[34]:


with open('Basic_project_pickle','wb') as file:
    pickle.dump(reg,file)


# ### we can still improve the quality of models by cleaning data and crossvalidating various models and also hyper parameter tuning as this is my first data science i am gonna end this small and simple project here 

# In[ ]:




