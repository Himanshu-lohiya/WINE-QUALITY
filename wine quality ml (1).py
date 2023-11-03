#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\wineQT.csv')
df.head()
df.tail()


# In[ ]:


df["quality"].value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.pop("Id")


# In[ ]:


X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = df['quality']


# 

# In[ ]:


col=df.columns
print(col)
sns.pairplot(df,x_vars=X = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],y_vars='quality', kind="scatter"
)


# In[ ]:


sns.pairplot(df, hue="quality")


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_LR=LinearRegression()
model_LR.fit(X_train,y_train)


# In[ ]:


y_pred = model_LR.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

r2 = r2_score(y_test, y_pred)

print("R-squared (R2) Score:", r2)


# In[ ]:


x.columns


# In[ ]:


model_LR.predict([[8,0.5,1.9]])


# In[ ]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Wine Quality")
plt.ylabel("Predicted Wine Quality")
plt.title("Actual vs. Predicted Wine Quality")
plt.show()


# In[ ]:




