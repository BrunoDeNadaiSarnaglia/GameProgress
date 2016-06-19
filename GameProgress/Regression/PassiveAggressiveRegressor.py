__author__ = 'Bruno'
__author__ = 'Bruno'

import numpy
from pandas import DataFrame
from sklearn import preprocessing, decomposition
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet, PassiveAggressiveRegressor
from sklearn.pipeline import Pipeline

__author__ = 'Bruno'

class PassiveAggressiveRegression(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Passive Aggressive Regression"
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('passiveAggressive', PassiveAggressiveRegressor())
        ])
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=[1, 2, 4, 6, 8, 10], passiveAggressive__C=numpy.logspace(0, 4, 2), passiveAggressive__loss=['epsilon_insensitive', 'squared_epsilon_insensitive']), scoring='r2')
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        print grid_search.best_params_
        print grid_search.grid_scores_
        print "accuracy: " + str(acc)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
