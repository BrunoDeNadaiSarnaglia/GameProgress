from pandas import DataFrame
from sklearn import decomposition, naive_bayes
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score

__author__ = 'Bruno'

class NaiveBayesianFitter(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "Naive Bayesian"
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA()),
            ('ridge', naive_bayes.GaussianNB())
        ])
        n_components = [2, 4, 6, 8, 10, 12]
        grid_search = GridSearchCV(pipelineFit, dict(pca__n_components=n_components))
        grid_search.fit(X, y)
        acc = grid_search.best_score_
        y_pred = grid_search.best_estimator_.predict(X)
        print grid_search.grid_scores_
        print "accuracy: " + str(acc) + "   score: " + str(precision_score(y, y_pred))
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
