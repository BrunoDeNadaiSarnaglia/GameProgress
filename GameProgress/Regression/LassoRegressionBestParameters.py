__author__ = 'Bruno'
import numpy
from pandas import DataFrame
from sklearn import preprocessing, decomposition, cross_validation
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

__author__ = 'Bruno'

class LassoRegressionBestParameter(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Lasso"
        X = preprocessing.scale(X)
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA(n_components=4)),
            ('lasso', Lasso(alpha=0.1))
        ])
        score = cross_validation.cross_val_score(pipelineFit, X, y, cv=3, scoring='r2')
        print "Precision:"
        print score
        return cross_validation.cross_val_predict(pipelineFit, X, y, cv=3)

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
