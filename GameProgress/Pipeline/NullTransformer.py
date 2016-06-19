from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'

class NullTransformer(TransformerMixin):


    def transform(self, X, **transform_params):
        return DataFrame(X.apply(lambda x: self.transformNull(x)))

    def transformNull(self, x):
        xNaN = []
        for feature in x:
            if feature == 'NULL':
                xNaN.append('NaN')
            else:
                xNaN.append(feature)
        return xNaN

    def fit(self, X, y=None, **fit_params):
        return self