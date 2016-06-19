from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'


class ClassYTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        data = DataFrame(X[['completed', 'completed_post']])
        result = []
        for index, row in data.iterrows():
            result.append(self.GetClass(row))
        return DataFrame(result)

    def GetClass(self, x):
        if x[1] == 'NaN' or x[0] == 'NaN':
            return 0
        if x[1] > x[0]:
            return 1
        return 0

    def fit(self, X, y=None, **fit_params):
        return self
