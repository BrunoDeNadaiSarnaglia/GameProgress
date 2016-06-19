__author__ = 'Bruno'


import numpy
from pandas import DataFrame
from sklearn import preprocessing, decomposition, linear_model
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

__author__ = 'Bruno'

class BayesianRidgeRegression(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Bayesian Ridge"
        X = preprocessing.scale(X)
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('lasso', linear_model.BayesianRidge())
        ])
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=[1, 2, 4, 6, 8, 10]), scoring='r2')
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        print grid_search.best_params_
        print grid_search.grid_scores_
        print "r2: " + str(acc)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
