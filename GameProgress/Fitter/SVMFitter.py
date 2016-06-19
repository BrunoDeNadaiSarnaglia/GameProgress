import numpy
from pandas import DataFrame
from sklearn import decomposition, preprocessing
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score

__author__ = 'Bruno'



class SVMFitter(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "SVM rbf kernel"
        X = preprocessing.scale(X)
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('svc', SVC(kernel="rbf", max_iter=-1))
        ])
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=[1, 2, 4, 6, 8, 10], svc__C=numpy.logspace(-1, 2, 1)), scoring='accuracy')
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        y_pred = grid_search.best_estimator_.predict(X)
        print grid_search.grid_scores_
        print "accuracy: " + str(acc) + "   score: " + str(precision_score(y, y_pred))
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

