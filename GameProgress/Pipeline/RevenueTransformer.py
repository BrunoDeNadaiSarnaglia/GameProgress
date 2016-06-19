from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'


class RevenueTransformer(TransformerMixin):

    __columnName = None

    def __init__(self, columnName = 'revenue'):
        self.__columnName = columnName

    def transform(self, X, **transform_params):
        hours = DataFrame(X['revenue'].apply(lambda x: self.GetRevenue(x)))
        hours.columns = [self.__columnName]
        return hours

    def GetRevenue(self, x):
        if x == 'NaN':
            return 0
        return float(x)

    def fit(self, X, y=None, **fit_params):
        return self

