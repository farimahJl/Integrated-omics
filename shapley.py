"Author: Hicham Jemil and Maryam Jalali"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn import model_selection, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import shap
from IPython.display import display



class Shap():
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


        # x = np.linspace(0, 2 * np.pi, 400)
        # y = np.sin(x ** 2)

        # fig, axs = plt.subplots(2, 2, figsize=(30,20))
        # axs[0, 0].plot(x, y)
        # axs[0, 0].set_title('Axis [0, 0]')
        # axs[0, 1].plot(x, y, 'tab:orange')
        # axs[0, 1].set_title('Axis [0, 1]')
        # axs[1, 0].plot(x, -y, 'tab:green')
        # axs[1, 0].set_title('Axis [1, 0]')
        # axs[1, 1].plot(x, -y, 'tab:red')
        # axs[1, 1].set_title('Axis [1, 1]')

        # for ax in axs.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')

        # # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()

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
    


