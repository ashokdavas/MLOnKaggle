#Iris dataset visualization.
#Iris dataset is built in sklearn

from sklearn import datasets
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the data and some metadata
iris_data = datasets.load_iris()
print iris_data.feature_names
x_data = iris_data.data
y_data = iris_data.target
species_names = iris_data.target_names
attribute_names = ["sepal_length","sepal_width","petal_length","petal_width"]

#check the distribution on the basis of sepal_length

# plt.scatter(y_data, x_data[:,0])
# plt.show()

# plt.scatter(x_data[:,1],y_data)
# plt.show()
#
# plt.scatter(x_data[:,2],y_data)
# plt.show()
#
# plt.scatter(x_data[:,3],y_data)
# plt.show()
markers = {'*': 'star', '2': 'tri_up', 2: 'tickup', 'o': 'circle', 4: 'caretleft', 5: 'caretright', '_': 'hline', '.': 'point', 'd': 'thin_diamond', '4': 'tri_right', '': 'nothing', 'None': 'nothing', 3: 'tickdown', ' ': 'nothing', 7: 'caretdown', 'x': 'x', 0: 'tickleft', '+': 'plus', '<': 'triangle_left', '|': 'vline', '8': 'octagon', 1: 'tickright', 6: 'caretup', 's': 'square', 'p': 'pentagon', ',': 'pixel', '^': 'triangle_up', 'D': 'diamond', None: 'nothing', 'H': 'hexagon2', '3': 'tri_left', '>': 'triangle_right', 'h': 'hexagon1', 'v': 'triangle_down', '1': 'tri_down'}
colors = np.random.rand(150)
# print colors.shape
colors = colors.reshape((150,1))
color = ['red', 'blue', 'lightgreen']
mark = ['_','o','*']
# print y_data
# # cmap = np.empty(type="string")
# print y_data.shape
# print y_data[0]
# for i in range(150):
#     cmap[i] = color[y_data[i]]
cmap = [color[y_data[i]] for i in range(150)]
def map_color(id):
    # colr = [color[id] for x in y_data if x == id]
    # return colr
    colr = []
    for x in y_data:
        if x == id:
            # print x
            colr.append(color[id])
    return colr
mmap = [mark[y_data[i]] for i in range(150)]
cmap = map_color(0)
print cmap
# cmap = np.reshape(cmap,newshape=())
# plt.scatter(x_data[:,3],x_data[:,2],c=cmap,alpha=1.0,marker=mmap[0])
# plt.show()

# for idx,label in enumerate(species_names):
#     plt.scatter(x_data[y_data == idx,3], x_data[y_data == idx,2],c=map_color(idx),marker=mark[idx])
#
# plt.show()
#
# for idx,label in enumerate(species_names):
#     plt.scatter(x_data[y_data == idx,0], x_data[y_data == idx,1],c=map_color(idx),marker=mark[idx])
#
# plt.show()
#
# for idx,label in enumerate(species_names):
#     plt.scatter(x_data[y_data == idx,0], x_data[y_data == idx,2],c=map_color(idx),marker=mark[idx])
#
# plt.show()
#
# for idx,label in enumerate(species_names):
#     plt.scatter(x_data[y_data == idx,3], x_data[y_data == idx,1],c=map_color(idx),marker=mark[idx])
#
# plt.show()

# # can't use sns as sns demands single data
# sns.jointplot(x= 0 ,y= 1 ,data=x_data)

# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

x_train,x_test,y_training,y_test = train_test_split(x_data,y_data,test_size=0.2)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#Can provide values to SVC e.g. kernel, gamma,c etc
svm = SVC()
svm.fit(x_train_std,y_training)
svm.score(x_test_std,y_test)
z = svm.predict(x_test_std)
print z.shape,x_test_std.shape
print z.shape[0]
z = z.reshape((z.shape[0],1))

color_cf = ["cyan","magenta","yellow"]
cmap = ListedColormap(color_cf[:len(np.unique(y_test))])
color_complete = [color_cf[id] for id in y_data]
print color_complete

plt.figure()
for idx,cl in enumerate(species_names):
    # print x_test_std[y_test == idx, 0].shape,x_test_std[y_test == idx,1].shape,z[y_test ==idx,0].shape
    x = x_test_std[y_test == idx, 2]
    x = x.reshape((x.shape[0],1))
    y = x_test_std[y_test == idx,3]
    y = y.reshape((y.shape[0],1))
    zx = z[y_test == idx,0]
    zx = zx.reshape((zx.shape[0],1))
    print x,zx
    print x.shape, y.shape, zx.shape, "idx: ", idx
    plt.contourf(x,y,zx)
    # plt.title('Simplest default with labels')
    # plt.scatter(x,y, c=map_color(idx), marker=mark[idx],s=100)
plt.show()
#extrapolate test data as it is very less in size.


