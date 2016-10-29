# API reference

# Distribution plots
# jointplot(x, y[, data, kind, stat_func, ...]) 	Draw a plot of two variables with bivariate and univariate graphs.
# pairplot(data[, hue, hue_order, palette, ...]) 	Plot pairwise relationships in a dataset.
# distplot(a[, bins, hist, kde, rug, fit, ...]) 	Flexibly plot a univariate distribution of observations.
# kdeplot(data[, data2, shade, vertical, ...]) 	Fit and plot a univariate or bivariate kernel density estimate.
# rugplot(a[, height, axis, ax]) 	Plot datapoints in an array as sticks on an axis.


# Regression plots
# lmplot(x, y, data[, hue, col, row, palette, ...]) 	Plot data and regression model fits across a FacetGrid.
# regplot(x, y[, data, x_estimator, x_bins, ...]) 	Plot data and a linear regression model fit.
# residplot(x, y[, data, lowess, x_partial, ...]) 	Plot the residuals of a linear regression.
# interactplot(x1, x2, y[, data, filled, ...]) 	Visualize a continuous two-way interaction with a contour plot.
# coefplot(formula, data[, groupby, ...]) 	Plot the coefficients from a linear model.

# Categorical plots
# factorplot([x, y, hue, data, row, col, ...]) 	Draw a categorical plot onto a FacetGrid.
# boxplot([x, y, hue, data, order, hue_order, ...]) 	Draw a box plot to show distributions with respect to categories.
# violinplot([x, y, hue, data, order, ...]) 	Draw a combination of boxplot and kernel density estimate.
# stripplot([x, y, hue, data, order, ...]) 	Draw a scatterplot where one variable is categorical.
# swarmplot([x, y, hue, data, order, ...]) 	Draw a categorical scatterplot with non-overlapping points.
# pointplot([x, y, hue, data, order, ...]) 	Show point estimates and confidence intervals using scatter plot glyphs.
# barplot([x, y, hue, data, order, hue_order, ...]) 	Show point estimates and confidence intervals as rectangular bars.
# countplot([x, y, hue, data, order, ...]) 	Show the counts of observations in each categorical bin using bars.

# Matrix plots
# heatmap(data[, vmin, vmax, cmap, center, ...]) 	Plot rectangular data as a color-encoded matrix.
# clustermap(data[, pivot_kws, method, ...]) 	Plot a hierarchically clustered heatmap of a pandas DataFrame
# Timeseries plots
# tsplot(data[, time, unit, condition, value, ...]) 	Plot one or more timeseries with flexible representation of uncertainty.
# Miscellaneous plots
# palplot(pal[, size]) 	Plot the values in a color palette as a horizontal array.

# Axis grids
# FacetGrid(data[, row, col, hue, col_wrap, ...]) 	Subplot grid for plotting conditional relationships.
# PairGrid(data[, hue, hue_order, palette, ...]) 	Subplot grid for plotting pairwise relationships in a dataset.
# JointGrid(x, y[, data, size, ratio, space, ...]) 	Grid for drawing a bivariate plot with marginal univariate plots.