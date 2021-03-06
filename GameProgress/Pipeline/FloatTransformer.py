from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'


class FloatTransformer(TransformerMixin):

    __columnName = None

    def __init__(self, columnName):
        self.__columnName = columnName

    def transform(self, X, **transform_params):
        hours = DataFrame(X[self.__columnName].apply(lambda x: self.GetFloat(x)))
        hours.columns = [self.__columnName]
        return hours

    def GetFloat(self, x):
        if x == 'NaN':
            return 'NaN'
        return float(x)

    def fit(self, X, y=None, **fit_params):
        return self

