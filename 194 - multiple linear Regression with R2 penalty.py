#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #works with multidimensional arrays
import pandas as pd #format the data into columns and rows
import matplotlib.pyplot as plt #2d visualization
import statsmodels.api as sm #summaries
import seaborn #nice graphs
seaborn.set()


# In[2]:


data = pd.read_csv('1.02. Multiple linear regression.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[7]:


y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]


# In[8]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()


# In[9]:


results.summary()


# In[ ]:




