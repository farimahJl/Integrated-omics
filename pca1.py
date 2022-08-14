#!/usr/bin/env python

""" This module contains functionality to perform PCA and plot a scatter plot.

Example usage:
run_pca:
    principal_df, pca = run_pca(x, n_components=2)

plot_scatter:
    plot_scatter(final_df, final_df['target'],
        0, 1,
        '2 Component PCA', targets, colors)

This was performed on the BMI dataset and the result are shown below:
Together, the first two principal components contain 95.80% of the information.
The first principal component contains 72.77% of the variance and
the second principal component contains 23.03% of the variance.
The third and fourth principal component contained the rest of the variance of the dataset.

What are other applications of PCA (other than visualizing data)?
If your learning algorithm is too slow because the input dimension is too high,
then using PCA to speed it up is a reasonable choice. (most common application in my opinion).
We will see this in the MNIST dataset.
If memory or disk space is limited,
PCA allows you to save space in exchange for losing a little of the data's information.
This can be a reasonable tradeoff.

What are the limitations of PCA?
PCA is not scale invariant. check: we need to scale our data first.
The directions with largest variance are assumed to be of the most interest
Only considers orthogonal transformations (rotations) of the original variables
PCA is only based on the mean vector and covariance matrix.
Some distributions (multivariate normal) are characterized by this, but some are not.
If the variables are correlated, PCA can achieve dimension reduction. If not,
PCA just orders them according to their variances.
"""


__author__ = "Maryam Jalali"
__license__ = "MIT"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(X, n_components=2):
    """Runs principal component analysis on a dataset.

    Returns a dataframe with principal components
    and the pca object.

    Keyword arguments:
    X -- Feature matrix
    n_components -- number of components
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components)

    return principal_df, pca

def plot_scatter(X, y, col1, col2, title, targets, colors):
    """Plots a scatter matrix of 2 columns.

    Keyword arguments:
    X -- feature matrix
    y -- target series
    col1 -- name of first column
    col2 -- name of second column
    title -- title for the figure
    targets -- targets to compare with y
    colors -- colors to give the targets
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(col1, fontsize=15)
    ax.set_ylabel(col2, fontsize=15)
    ax.set_title(title, fontsize=20)

    # map colors
    for target, color in zip(targets, colors):
        indices_to_keep = y == target
        ax.scatter(X.loc[indices_to_keep, col1],
            X.loc[indices_to_keep, col2],
            c=color,
            s=50)

    ax.legend(targets)
    ax.grid()

    plt.show()


if __name__ == '__main__':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # loading dataset into Pandas DataFrame
    df = pd.read_csv(url
                    , names=['sepal length','sepal width','petal length','petal width','target'])

    # Standardize the Data
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    x = df.loc[:, features].values
    y = df.loc[:, ['target']].values
    x = StandardScaler().fit_transform(x)

    # PCA Projection to 2D
    principal_df, pca = run_pca(x, n_components=2)

    final_df = pd.concat([principal_df, df[['target']]], axis = 1)

    # Visualize 2D Projection
    # Use a PCA projection to 2d to visualize the entire data set.
    # You should plot different classes using different colors or shapes.
    # Do the classes seem well-separated from each other?
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    plot_scatter(final_df, final_df['target'],
        0, 1,
        '2 Component PCA', targets, colors)

    # The explained variance tells us how much information (variance)
    # can be attributed to each of the principal components.
    print('Explained variance ratio:')
    print(pca.explained_variance_ratio_)
