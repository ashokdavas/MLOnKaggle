import sklearn as skt
from sklearn import datasets,svm,linear_model,neighbors,naive_bayes,gaussian_process,tree,ensemble,multiclass,discriminant_analysis
from sklearn import cross_validation, model_selection
import numpy as np

iris_data = datasets.load_iris()
digits_data = datasets.load_digits()
boston_data = datasets.load_boston()
cancer_data = datasets.load_breast_cancer()
diabetes_data = datasets.load_diabetes()

print iris_data.data.shape
print digits_data.data.shape
print boston_data.data.shape
print cancer_data.data.shape
print diabetes_data.data.shape

data_x = iris_data.data
data_y = iris_data.target
iris_training_x, iris_test_x, iris_training_y, iris_test_y = model_selection.train_test_split(data_x,data_y,test_size=0.1)
print data_x.shape
print data_x.dtype
# print np.random.choice(np.arange(150),size=140)
# iris_training = data_x[0:140]
# iris_training_y= data_y[0:140]
# iris_test = data_x[140:]
# iris_test_y = data_y[140:]
# print  iris_test
# iris_test = data_x
linear_svc = svm.LinearSVC()
svc_kernel = svm.SVC()

log_reg = linear_model.LogisticRegression()
softmax_clf = linear_model.LogisticRegression(multi_class='multinomial')
sgd = linear_model.SGDClassifier()

neighbors_clf = neighbors.KNeighborsClassifier()
radius_neighbors_clf = neighbors.RadiusNeighborsClassifier()

naive_bayes_clf = naive_bayes.GaussianNB()

# gaussian_process_clf = gaussian_process.GaussianProcess()

dst_clf = tree.DecisionTreeClassifier()

rf_clf = ensemble.RandomForestClassifier()

ldf_clf = discriminant_analysis.LinearDiscriminantAnalysis()
qda_clf = discriminant_analysis.QuadraticDiscriminantAnalysis()

ldf_clf.fit(iris_training_x,iris_training_y)
print "lda: ",ldf_clf.score(iris_test_x,iris_test_y)
print ldf_clf.predict(iris_test_x)
print iris_test_y

qda_clf.fit(iris_training_x,iris_training_y)
print "qda: ",qda_clf.score(iris_test_x,iris_test_y)
print qda_clf.predict(iris_test_x)
print iris_test_y

rf_clf.fit(iris_training_x,iris_training_y)
print "random forest: ",rf_clf.score(iris_test_x,iris_test_y)
print rf_clf.predict(iris_test_x)
print iris_test_y

dst_clf.fit(iris_training_x,iris_training_y)
print "dicision tree: ",dst_clf.score(iris_test_x,iris_test_y)
print dst_clf.predict(iris_test_x)
print iris_test_y

naive_bayes_clf.fit(iris_training_x,iris_training_y)
print "naive bayes: ",naive_bayes_clf.score(iris_test_x,iris_test_y)
print naive_bayes_clf.predict(iris_test_x)
print iris_test_y

neighbors_clf.fit(iris_training_x,iris_training_y)
print "nearest neighbour: ",neighbors_clf.score(iris_test_x,iris_test_y)
print neighbors_clf.predict(iris_test_x)
print iris_test_y

sgd.fit(iris_training_x,iris_training_y)
print "sgd classifier: ",sgd.score(iris_test_x,iris_test_y)
print sgd.predict(iris_test_x)
print iris_test_y
#
# softmax_clf.fit(iris_training_x,iris_training_y)
# print "softmax classifier: ",softmax_clf.score(iris_test_x,iris_test_y)
# print softmax_clf.predict(iris_test_x)
# print iris_test_y

log_reg.fit(iris_training_x,iris_training_y)
print "logistic regression: ",log_reg.score(iris_test_x,iris_test_y)
print log_reg.predict(iris_test_x)
print iris_test_y

linear_svc.fit(iris_training_x,iris_training_y)
print "linear svc: ",linear_svc.score(iris_test_x,iris_test_y)
print linear_svc.predict(iris_test_x)
print iris_test_y

svc_kernel.fit(iris_training_x,iris_training_y)
print "svc kernel: ",svc_kernel.score(iris_test_x,iris_test_y)
print svc_kernel.predict(iris_test_x)
print iris_test_y