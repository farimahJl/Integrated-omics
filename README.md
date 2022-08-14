# Integrated-omics
This README explains what I did for the project and what code I wrote. It will first give a general overview of the *pipeline*, will go over *preprocessing* shortly, and then goes into the code.
for some parts of pca code I used (https://medium.com/search?q=pca)


# Pipeline
A schematic overview can be seen in figure 1. The yellow symbol indicates user input, the red square indicates a process that is performed by the application, and the green symbol indicates visualization.
https://github.com/RikvdPol/IntergratedOmics
![alt text](imgs/Flowchart.png)

The pipeline works as follows:
1. Read the dataset and define a target column, *y*.
2. Drop feature columns if specified.
3. Choose taxanomy, such as *species*.
4. Perform PCA.
5. Train model.
6. Perform shapely.
7. Display plots to user.

The user can interpret the plots and find the features that contribute most

# Preprocess
The data is preprocessed by using PCA (after scaling). PCA is a dimensionality reduction technique that tries to explain most variance. The data can also be preprocessed by using the *fillna* method to fill *NaN* values.

# Code
## PCA
The code for PCA can be found in `pca.py`.

Pipeline:
1. Define your features, *X*.
2. Scale your features by using the `StandardScaler`.
3. Use `pca` and it will return the principal components.

The components can be used for further analysis, such as analyzing the loadings, or training model.

The most important code is the following:

```python

def pca(x, n_components=2):
    pca = PCA(n_components)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
    return principalDf

```

Example:

`principalDf = pca(x, n_components=2)`

Advantages:
- If the algorithm cannot handle high dimensional data efficiently, we can reduce dimensions.

Disadvantages:
- Data should be scaled first, except coordinate data.

## Fill NaN
The code to fill *NaN* values can be found in `fillna.py`.

The code fills *NaN* values in a DataFrame with the same value.

Example usage:
```python

df = fillna(df_with_NaN, value)

```

## Random Forest Classifier
The Random Forest Classifier can be found in `rfc.py`.

Used to classify labels, given a set of features. It is a ensemble learning method by creating multiple decision trees. The output is the label that is selected by most decision trees.

It requires `features` and `labels`. It inherits from `AlgorithmBaseClass` which can be found on the original github.

Example usage:
```python

algorithm = RF(X, y)
clf, cv = algorithm.define_model()
algorithm.evaluate_model(clf, cv)

```

## XGBoost Classifier
The XGBoost classifier can be found in `xgbc.py`. It is a tree boosting algorithm that performs very well. It is often the best performing machine learning model and is used frequently at competitions.

This class uses the exact same layout as the Random Forest Classifier, as it also inherits `AlgorithmBaseClass`.

Example usage:
```python

algorithm = XGBoostClassifier(X, y)
clf, cv = algorithm.define_model()
algorithm.evaluate_model(clf, cv)

```

## Shapley plots
By using the `Shap` class in `shapley.py` we can plot the shapley values. These can then be interpreted by the researchers. 

The code was written by me and Hicham, but I'll only show my code.

Example usage of my code:

```python

shap = Shap(X_train, model)
shap.plot_waterfall(0)
shap.plot_force()
shap.plot_beeswarm()
shap.plot_heatmap()
shap.plot_scatter()

```

