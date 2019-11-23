#!/usr/bin/env python
# coding: utf-8

# In[46]:


from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import pandas as pd


# # Finding most influencial feature of boston data set

# In[47]:


def most_influencial():
    print("=======================Part 1=======================")
    boston_data = load_boston()
    regressor = linear_model.LinearRegression()
    regressor.fit(boston_data.data, boston_data.target)
    coef_df = pd.DataFrame(regressor.coef_,boston_data.feature_names, columns = ["Coefficient"])
    print('Most influencial factor is: \n{}.'
          .format(coef_df.loc[np.abs(coef_df['Coefficient']) == max(np.abs(coef_df['Coefficient']))].iloc[0]))


# # Comparing cluster size impacts in K-means clustering 

# In[48]:


def kmeansit():
    print("=======================Part 2=======================")
    iris_data = load_iris()
    squaredistance = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters = i).fit(iris_data.data)
        squaredistance.append(kmeans.inertia_) 
    plt.plot([i for i in range(1, 15)], squaredistance, '-o')
    plt.xlabel('Cluster count')
    plt.ylabel('Squared Distance')
    plt.savefig('kmeans_iris.png')
    plt.show()


# In[49]:


if __name__ == '__main__':
    most_influencial()
    kmeansit()


# In[ ]:




