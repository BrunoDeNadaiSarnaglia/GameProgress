from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'


class RepeaterTransformer(TransformerMixin):

    def __init__(self, columnNames):
        self.__columnNames = columnNames

    def transform(self, X, **transform_params):
        return DataFrame(X[self.__columnNames])

    def GetRevenue(self, x):
        if x == 'NULL':
            return 0
        return int(x)

    def fit(self, X, y=None, **fit_params):
        return self

