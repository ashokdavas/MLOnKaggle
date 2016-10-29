import numpy as np
import pandas as pd
from sklearn import datasets,model_selection
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("data/Iris.csv")
print iris.columns

#this let's us see what the data look like when segmented by one or more variables.
# The easiest way to do this is thorugh factorplot.
# Let's say that we we're interested in how cars' MPG has varied over time.
# Not only can we easily see this in aggregate:
#col devides the plot on the basis of given attribute and show them in columns
sns.factorplot(x="",y="",data=iris,row="",col="",hue=None)

#Let's say that we wanted to see KDE plots of the MPG distributions, separated by country of origin:
# g = sns.FacetGrid(df, col="origin")
# g.map(sns.distplot, "mpg")

#g = sns.FacetGrid(df, col="origin")
# g.map(plt.scatter, "horsepower", "mpg")

# g = sns.FacetGrid(df, col="origin")
# g.map(sns.regplot, "horsepower", "mpg")

#
# g = sns.pairplot(df[["mpg", "horsepower", "weight", "origin"]], hue="origin", diag_kind="hist")
# for ax in g.axes.flat:
#     plt.setp(ax.get_xticklabels(), rotation=45)


#As FacetGrid was a fuller version of factorplot,
# so PairGrid gives a bit more freedom on the same idea as pairplot by letting you control the individual plot types separately
# g = sns.PairGrid(df[["mpg", "horsepower", "weight", "origin"]], hue="origin")
# g.map_upper(sns.regplot)
# g.map_lower(sns.residplot)
# g.map_diag(plt.hist)
# for ax in g.axes.flat:
#     plt.setp(ax.get_xticklabels(), rotation=45)
# g.add_legend()
# g.set(alpha=0.5)

#  jointplot and JointGrid; these features let you easily view both a joint distribution and its marginals at once.
# sns.jointplot("mpg", "horsepower", data=df, kind='kde')

# g = sns.JointGrid(x="horsepower", y="mpg", data=df)
# g.plot_joint(sns.regplot, order=2)
# # g.plot_marginals(sns.distplot)
# Definition of a marginal distribution = If X and Y are discrete random variables and f (x,y) is the value of
# their joint probability distribution at (x,y), the functions given by:
# g(x) = Σy f (x,y) and h(y) = Σx f (x,y) are the marginal distributions of X and Y , respectively.

#Plotting univariate distributions
# x = np.random.normal(size=100)
# sns.distplot(x); # By default, this will draw a histogram and fit a kernel density estimate (KDE).
# sns.distplot(x, kde=False, rug=True);
# sns.distplot(x, hist=False, rug=True);
# sns.kdeplot(x, shade=True); # shade is for filling
##Fitting parametric distributions
# x = np.random.gamma(6, size=200)
# sns.distplot(x, kde=False, fit=stats.gamma);

# Plotting bivariate distributions
# sns.jointplot(x="x", y="y", data=df); #scatter plot by default
# sns.jointplot(x="x", y="y", data=df, kind="kde"); #joint distribution








# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5,kind="scatter")
# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
#
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)



# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(iris, hue="Species", size=6) \
   .map(sns.kdeplot, "PetalLengthCm") \
   .add_legend()

# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
#
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)



# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")


# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))


sns.plt.show()