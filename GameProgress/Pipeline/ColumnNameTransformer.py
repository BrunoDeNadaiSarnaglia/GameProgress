from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'

class ColumnNameTransformer(TransformerMixin):

    __columnNames = None

    def __init__(self, columnNames = []):
        self.__columnNames = columnNames

    def transform(self, X, **transform_params):
        dataFrame = DataFrame(X)
        dataFrame.columns = self.__columnNames
        return dataFrame

    def fit(self, X, y=None, **fit_params):
        return self
