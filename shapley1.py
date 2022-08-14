"""This module provides functionality to show the shap values.
The Shap class and shap_test function were written by Hicham.
The Shap.plot_Z functions were written by Maryam.

Example usage:
Shap:
    shap = Shap()
    shap.plot_force()

Warning: not all models are supported by the Shapley library.
"""

__author__ = ["Hicham Jemil", "Maryam Jalali"]
__license__ = "MIT"

import shap
from IPython.display import display



class Shap():
    """Creates Shap plots to be interpeted. Shows the most important features.

    Keyword arguments:
    X_train -- training features
    model -- model to train on data
    """
    def __init__(self, X_train, model):
        self.X_train = X_train
        self.model = model

        # model should be fit, this only works if the model is supported by the Shapley model
        try:
            self.explainer = shap.Explainer(model, X_train)
        except Exception as e:
            print(e)


    def shap_test(self):
        "Provides the user both local and global shap values based on the Game theory"
        shap.initjs()
        X_sampled = self.X_train.sample(100, random_state=10)  #in order to improve computing efficiency, a random v is chosen
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sampled)
        print("## Contributing features to diviate from the base value")
        print("Features in red contribute to a higher prediction")
        print("Features in blue contribute to a lower prediction")
        display(shap.force_plot(explainer.expected_value, shap_values[0,:], X_sampled.iloc[0,:]))

        print("\n## Contributing effect of a single feature vs the model output")
        print("Shap values represents a feature's responsability for a change in a selected output")
        print("Vertical dispersion represents the interaction vs the other features")
        display(shap.force_plot(explainer.expected_value, shap_values, self.X_train))

        print("\n## Mean absolute contribution for each feature")
        display(shap.summary_plot(shap_values, X_sampled))

        print("\n## Mean absolute contribution for each feature")
        display(shap.summary_plot(shap_values, X_sampled, plot_type="bar"))

    def plot_waterfall(self, idx, max_display=10):
        """Create a waterfall plot"""
        shap_values = self.explainer(self.X_train)
        display(shap.plots.waterfall(shap_values[idx], max_display=max_display))

    def plot_force(self, idx=None, max_display=10):
        """Create a bar plot"""
        shap_values = self.explainer(self.X_train)
        if idx:
            display(shap.plots.bar(shap_values[idx], max_display=max_display))
        else:
            display(shap.plots.bar(shap_values, max_display=max_display))

    def plot_beeswarm(self, max_display=10):
        """Create beeswarm plot"""
        shap_values = self.explainer(self.X_train)
        display(shap.plots.beeswarm(shap_values, max_display=max_display))

    def plot_heatmap(self, max_display=10):
        """Create heatmap"""
        shap_values = self.explainer(self.X_train)
        display(shap.plots.heatmap(shap_values, max_display=max_display))

    def plot_scatter(self, col):
        """Create scatterplot"""
        shap_values = self.explainer(self.X_train)
        display(shap.plots.scatter(shap_values[:, col], color=shap_values))



def script_shapley(X_train, model):
    Shap(X_train, model).shap_test()
