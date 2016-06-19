from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer

__author__ = 'Bruno'



class ImputerTransformer(TransformerMixin):

    def __init__(self, columnName, strategy):
        self.__columnName = columnName
        self.__strategy = strategy

    def transform(self, X, **transform_params):
        data = DataFrame(X[self.__columnName])
        imputer = Imputer(missing_values='NaN', strategy=self.__strategy)
        imputer.fit(data, None)
        return imputer.transform(data)

    def fit(self, X, y=None, **fit_params):
        return self