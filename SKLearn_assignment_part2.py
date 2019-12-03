
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# # Comparing cluster size impacts in K-means clustering 


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



if __name__ == '__main__':
    kmeansit()
    
