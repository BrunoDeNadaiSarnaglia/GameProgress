import numpy
from pandas import DataFrame
from sklearn import decomposition, preprocessing
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.metrics import precision_score

__author__ = 'Bruno'

class SVRRegression(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "SVR rbf kernel"
        X = preprocessing.scale(X)
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('svc', SVR(kernel="rbf", max_iter=-1))
        ])
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=[4, 6, 8, 10], svc__C=numpy.logspace(-1, 1, 3), svc__gamma=numpy.logspace(-2, 2, 5)), scoring='r2')
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        print grid_search.best_params_
        print grid_search.grid_scores_
        print "r2: " + str(acc)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
