from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'



class DeviceEncoderTransformer(TransformerMixin):

    __fieldMap = {'ipod': 0.0, 'iphone': 1.0, 'ipad': 2.0}

    def __init__(self, columnName='device'):
        self.__columnName = columnName

    def transform(self, X, **transform_params):
        hours = DataFrame(X[self.__columnName].apply(lambda x: self.__fieldMap[x]))
        hours.columns = [self.__columnName]
        return hours

    def fit(self, X, y=None, **fit_params):
        return self
