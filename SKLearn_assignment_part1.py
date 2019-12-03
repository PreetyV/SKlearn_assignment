
from sklearn.datasets import load_boston
from sklearn import linear_model
import numpy as np
import pandas as pd


# # Finding most influencial feature of boston data set



def most_influencial():
    print("=======================Part 1=======================")
    boston_data = load_boston()
    regressor = linear_model.LinearRegression()
    regressor.fit(boston_data.data, boston_data.target)
    coef_df = pd.DataFrame(regressor.coef_,boston_data.feature_names, columns = ["Coefficient"])
    print('Most influencial factor is: \n{}.'
          .format(coef_df.loc[np.abs(coef_df['Coefficient']) == max(np.abs(coef_df['Coefficient']))].iloc[0]))



if __name__ == '__main__':
    most_influencial()
    
