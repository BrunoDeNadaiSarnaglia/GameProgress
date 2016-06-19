from sklearn.ensemble import RandomForestRegressor

__author__ = 'Bruno'

import numpy
from pandas import DataFrame
from sklearn import decomposition, preprocessing
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn import neighbors

__author__ = 'Bruno'



class NearestNeighborsFitter(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Nearest Neighbors"
        X = preprocessing.scale(X)
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('nearestNeighbors', neighbors.KNeighborsClassifier())
        ])
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=[1, 2, 4, 6, 8, 10], nearestNeighbors__n_neighbors=[1, 5, 10, 20, 30, 40, 50, 60, 70]), scoring='accuracy')
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        y_pred = grid_search.best_estimator_.predict(X)
        print grid_search.grid_scores_
        print "accuracy: " + str(acc) + "   score: " + str(precision_score(y, y_pred))
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

