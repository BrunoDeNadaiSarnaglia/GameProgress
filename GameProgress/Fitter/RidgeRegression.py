__author__ = 'Bruno'


import numpy
from pandas import DataFrame
from sklearn import decomposition, linear_model
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline

__author__ = 'Bruno'


class RidgeRegressionFitter(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Ridge Regression"
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('ridge', linear_model.Ridge())
        ])
        n_components = [2, 4, 6, 8, 10, 12]
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=n_components, ridge__alpha=numpy.logspace(-2,5,2)))
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        y_pred = grid_search.best_estimator_.predict(X)
        print grid_search.grid_scores_
        print "accuracy: " + str(acc)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

