from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from AlgorithmBaseClass import AlgorithmBaseClass
import sys
import Logging


# paste this code under the other class i suppose
class RF(AlgorithmBaseClass):
    """Create a random forest classifier to classify observations.
    
    Inherits from the AlgorithmBaseClass and only implements the `define_model` function.

    Keyword arguments:
    file -- the features
    labelname -- the target
    """
    
    def define_model(self, n_splits=10, n_repeats=3, random_state=None):
        """Define a the random forest classifier with cross validation and return it."""
        logs = Logging.Logging()

        try:
            clf = RandomForestClassifier()
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            msg = f'Model {type(clf)} definition succeeded.'
            logs.create_logs(self.__class___.__name__, msg)
            return clf, cv
        except Exception as e:
            msg = f"{e}: Model definition failed"
            logs.create_logs(self.__class__.__name__, msg)
            sys.exit(0)
