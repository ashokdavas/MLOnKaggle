import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,StandardScaler

movies = pd.read_csv("data/movie_metadata.csv")
print (movies.shape)
print (movies.columns)

#drop columns which does not seem to have any effect on movie rating

#get numeric data for computation and correlation purposes
numerical_data = movies.select_dtypes(exclude=["object"])

# take out the y value(imdb_score from data)
score_imdb= numerical_data["imdb_score"]
numerical_data = numerical_data.drop(["imdb_score"],axis=1)
year_category = numerical_data["title_year"]
numerical_data = numerical_data.drop(["title_year"],axis=1)
numerical_columns = numerical_data.columns
# print (numerical_columns.shape)
# print (numerical_data[numerical_columns])
# print (numerical_data.describe())


#fill missing values and normalize the data
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)      #default values
numerical_data[numerical_columns] = imp.fit_transform(numerical_data[numerical_columns])
# print (numerical_data.describe())
# scaler = StandardScaler()
# numerical_data = scaler.fit_transform(numerical_data)
# print (numerical_data.describe())
# print (numerical_data.shape)
# numerical_data = pd.DataFrame(numerical_data)
# print (numerical_data.describe())

#get non_numeric informational content
information_data = movies.select_dtypes(include=["object"])
print (information_data.columns)


#numpy corrcoef returns symmetric metrics of correlation coef
#Use -from scipy.stats.stats import pearsonr   print pearsonr(a,b)
#check attributes for correlation with movie rating
low_covariance_1 = []
low_covariance_2 = []
low_covariance_15 = []
low_covariance_2g = []
for x in numerical_columns:
    z = (np.corrcoef(numerical_data[x],y=score_imdb))
    if(np.fabs(z[0,1]) < 0.1):
        low_covariance_1.append(x)
    elif(np.fabs(z[0,1]) < 0.15):
        low_covariance_15.append(x)
    elif(np.fabs(z[0,1])<0.2):
        low_covariance_2.append(x)
    else:
        low_covariance_2g.append(x)

print (low_covariance_2g,low_covariance_2,low_covariance_15,low_covariance_1)

from sklearn.feature_selection import SelectKBest,SelectPercentile,RFE,RFECV,SelectFromModel
from sklearn.svm import SVR,SVC
from sklearn.linear_model import Lasso
#data which has high correlation with imdb_score is selected
select_k = SelectKBest(k=8)
x_transformed = select_k.fit_transform(numerical_data,y=score_imdb) #x_transformed is numpy array not pandas
#sklearn returns numpy array not pandas object
print (x_transformed.shape)
# print (x_transformed.columns)
print (x_transformed[0,:])
print (numerical_data.head(1))

#The underlying estimator SVR has no `coef_` or `feature_importances_` attribute.
#  Either pass a fitted estimator to SelectFromModel or call fit before calling transform.
estimator = SVR(kernel="linear").fit(numerical_data,score_imdb)
rfe = SelectFromModel(estimator,prefit=True)
x_transformed = rfe.transform(numerical_data)
print (x_transformed[0,:])

#RFE use recursive selecting of attributes which is a time counsuming process.
estimator = SVR(kernel="linear")
selector = RFE(estimator)
selector = selector.fit(numerical_data,score_imdb)
print (selector.support_)
print (selector.ranking_)
x_transformed = selector.transform(numerical_data)
print (x_transformed[0,:])
#Fit a regression model on numeric data
from sklearn.model_selection import train_test_split


def svm_score(test_y, predict_y):
    # convert to numpy array to compare both predict and actual array
    # Iris_test_y contain indexes from dataframe(parent)
    iris_test_y = np.array(test_y)
    diff = 0
    total_size = test_y.shape[0]
    # print (total_size,test_y.iloc[0],predict_y[0])
    for idx in range(total_size):
        diff += np.fabs(test_y.iloc[idx]-predict_y[idx])
    return diff/total_size

def fit_model(model_to_print,model,x_data,y_data):
    training_x,test_x,training_Y,test_y = train_test_split(x_data,y_data,test_size=0.001)
    model.fit(X=training_x,y=training_Y)
    predicted_y = model.predict(test_x)
    print  (model_to_print,"training",model.score(training_x,training_Y))
    print  (model_to_print,model.score(test_x,test_y))
    # print (model_to_print,"training",svm_score(training_Y,model.predict(training_x)))
    # print (model_to_print,svm_score(test_y,predicted_y))

#On complete data without feature extraction
svr_model = SVR(kernel='rbf') #default
svr_linear_model = SVR(kernel="linear")
svr_poly_model = SVR(kernel="poly") #default degree is 3

plt.figure()
# plt.plot(score_imdb,label="original data")
fit_model("SVR rbf: ",svr_model,numerical_data,score_imdb)
fit_model("SVR linear: ",svr_model,numerical_data,score_imdb)
fit_model("SVR poly: ",svr_model,numerical_data,score_imdb)
plt.show()


#same model on transformed data with data selection
fit_model("transformed , svr rbf: ",svr_model,x_transformed,score_imdb)
fit_model("transformed , svr linear: ",svr_model,x_transformed,score_imdb)
fit_model("transformed , svr poly: ",svr_model,x_transformed,score_imdb)

#using knn regression
from sklearn.neighbors import KNeighborsRegressor

default_knn = KNeighborsRegressor(n_neighbors=5)
knn_10 = KNeighborsRegressor(n_neighbors=10)
knn_20 = KNeighborsRegressor(n_neighbors=20)

fit_model(" knn with k=5: ",default_knn,numerical_data,score_imdb)
fit_model(" knn with k=10: ",knn_10,numerical_data,score_imdb)
fit_model(" knn with k=20: ",knn_20,numerical_data,score_imdb)

#same model on transformed data with data selection
fit_model("transformed , knn with k=5: ",default_knn,x_transformed,score_imdb)
fit_model("transformed , knn with k=10: ",knn_10,x_transformed,score_imdb)
fit_model("transformed , knn with k=20: ",knn_20,x_transformed,score_imdb)

#Other regression models

from sklearn.linear_model import LinearRegression,Ridge

linear_reg = LinearRegression()
fit_model("linear regression: ",linear_reg,numerical_data,score_imdb)
fit_model("linear regression transformed: ",linear_reg,x_transformed,score_imdb)

#Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients.
#alpha is the rete of penalty
ridge_1 = Ridge(alpha=1.0)
ridget_point_5 = Ridge(alpha=0.5)
ridget_point_25 = Ridge(alpha=0.25)
fit_model("Ridge alpha =1:",ridge_1,numerical_data,score_imdb)
fit_model("Ridge alpha =0.5 :",ridge_1,numerical_data,score_imdb)
fit_model("Ridge alpha =0.25:",ridge_1,numerical_data,score_imdb)

fit_model("Ridge transformed alpha =1:",ridge_1,x_transformed,score_imdb)
fit_model("Ridge transformed alpha =0.5 :",ridge_1,x_transformed,score_imdb)
fit_model("Ridge transformed alpha =0.25:",ridge_1,x_transformed,score_imdb)

#By plotting the distribution against predicted values. You can see that values are in the middle range(5,7) and have a peak at 6.
#While the original distribution is more randomly distributed.
#Values of score returns the mean deviation from actual score.