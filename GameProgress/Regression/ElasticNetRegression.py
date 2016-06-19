__author__ = 'Bruno'

import numpy
from pandas import DataFrame
from sklearn import preprocessing, decomposition
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.pipeline import Pipeline

__author__ = 'Bruno'

class ElasticNetRegression(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Elastic Net"
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('elasticNet', ElasticNet(normalize=True))
        ])
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=[1, 2, 4, 6, 8, 10], elasticNet__alpha=numpy.logspace(-1, 1, 4)), scoring='r2')
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        print grid_search.best_params_
        print grid_search.grid_scores_
        print "r2: " + str(acc)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
