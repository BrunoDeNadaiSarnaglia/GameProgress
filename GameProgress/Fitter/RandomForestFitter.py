import numpy
from pandas import DataFrame
from sklearn import decomposition, linear_model
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline

__author__ = 'Bruno'


class RandomForestFitter(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Random Forest"
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('random', RandomForestClassifier())
        ])
        n_components = [2, 4, 6, 8, 10, 12]
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=n_components, random__n_estimators=[1,10,20]))
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        y_pred = grid_search.best_estimator_.predict(X)
        print grid_search.grid_scores_
        print "accuracy: " + str(acc)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

