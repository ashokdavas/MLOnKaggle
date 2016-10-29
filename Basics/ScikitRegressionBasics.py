import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model,neighbors,tree,svm,ensemble

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor