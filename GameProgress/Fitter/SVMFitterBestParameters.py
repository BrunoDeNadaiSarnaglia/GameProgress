from pandas import DataFrame
from sklearn import preprocessing, decomposition, cross_validation
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score


class SVMFitterBestParameters(TransformerMixin):

    def fit(self, X, y=None,*args, **kwargs):
        print "SVM rbf kernel"
        X = preprocessing.scale(X)
        pipelineFit = Pipeline([
            ('pca', decomposition.PCA(n_components=8)),
            ('svc', SVC(kernel="rbf", max_iter=-1, C=0.1))
        ])
        score = cross_validation.cross_val_score(pipelineFit, X, y, cv=3, scoring='precision')
        print "Precision:"
        print score
        score = cross_validation.cross_val_score(pipelineFit, X, y, cv=3, scoring='accuracy')
        print "Accuracy:"
        print score
        return cross_validation.cross_val_predict(pipelineFit, X, y, cv=3)

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

