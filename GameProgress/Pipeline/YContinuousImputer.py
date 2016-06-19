from pandas import DataFrame
from sklearn.base import TransformerMixin

__author__ = 'Bruno'


class YContinuousImputer(TransformerMixin):

    def transform(self, X, **transform_params):
        data = DataFrame(X[['completed', 'completed_post']])
        result = []
        for index, row in data.iterrows():
            result.append(self.GetClass(row))
        return DataFrame(result)

    def GetClass(self, x):
        if x[1] == 'NaN' and x[0] == 'NaN':
            return 0.5
        if x[0] == 'NaN':
            return x[1]
        if x[1] == 'NaN':
            return x[0]
        return x[1]

    def fit(self, X, y=None, **fit_params):
        return self