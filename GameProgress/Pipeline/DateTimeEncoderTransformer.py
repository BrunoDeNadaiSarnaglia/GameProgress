from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'


class DateTimeEncoderTransformer(TransformerMixin):

    __fieldMap = {'2015-05-25': 0.0, '2015-05-26': 1.0, '2015-05-27': 2.0}

    def __init__(self, columnName='ls_date'):
        self.__columnName = columnName

    def transform(self, X, **transform_params):
        hours = DataFrame(X[self.__columnName].apply(lambda x: self.__fieldMap.get(x, 'NaN')))
        hours.columns = [self.__columnName]
        return hours

    def fit(self, X, y=None, **fit_params):
        return self
