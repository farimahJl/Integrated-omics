"""This module facilitates an XGBoost classfier.
This can e.g. be used for protein classification.

Example usage:
RF:
    alg = XGBoostClassifier(...)
    clf, cv = alg.define_model()
    score = cross_val_score(clf, cv=cv)
    print(score)

For exact usage, see the AlgorithmBaseClass
"""

__author__ = "Maryam Jalali"
__license__ = "MIT"

import sys

import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from AlgorithmBaseClass import AlgorithmBaseClass
import Logging




# paste this code under the other class i suppose
class XGBoostClassifier(AlgorithmBaseClass):
    """Create a XGBoost classifier to classify observations.
    XGBoost is a boosting tree algorithm that is state of the art.

    Inherits from the AlgorithmBaseClass and only implements the `define_model` function.

    Keyword arguments:
    file -- the features
    labelname -- the target
    """

    def define_model(self, n_splits=10, n_repeats=3, random_state=None):
        """Define the XGBoost classifier with cross validation and return it."""
        logs = Logging.Logging()

        try:
            clf = XGBClassifier()
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            msg = f'Model {type(clf)} definition succeeded.'
            logs.create_logs(self.__class___.__name__, msg)
            return clf, cv
        except Exception as e:
            msg = f"{e}: Model definition failed"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)


if __name__ == '__main__':
    # testing
    data = pd.read_csv('Covariates.csv', sep='\t')
    data = data.dropna(axis=0)
    X = data[['Age', 'BMI']].values
    y = data['Gender'].values

    model= XGBoostClassifier()

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(scores)
