from Pipeline.ClassYTransformer import ClassYTransformer
from Pipeline.ColumnNameTransformer import ColumnNameTransformer
from Pipeline.DateTimeEncoderTransformer import DateTimeEncoderTransformer
from Pipeline.DeviceEncoderTransformer import DeviceEncoderTransformer
from Pipeline.FloatTransformer import FloatTransformer
from Pipeline.ImputerTransformer import ImputerTransformer
from Pipeline.IntTransformer import IntTransformer
from Pipeline.NullTransformer import NullTransformer
from Pipeline.RepeaterTransformer import RepeaterTransformer
from Pipeline.RevenueTransformer import RevenueTransformer
from Pipeline.UnitsTransformer import UnitsTransformer
from Pipeline.YContinuousImputer import YContinuousImputer

__author__ = 'Bruno'

__author__ = 'Bruno'
import numpy
from pandas import DataFrame
from sklearn import preprocessing, decomposition, cross_validation
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline, FeatureUnion

__author__ = 'Bruno'

class FinalModel(TransformerMixin):

    def __init__(self, classificationModel, regressionModel):
        self.classificationModel = classificationModel
        self.regressionModel = regressionModel

    def fit(self, D, y=None,*args, **kwargs):
        X = self.getX(D)
        y_cont = self.getYCont(D)
        y_desc = self.getYDesc(D)
        y_desc_pred = self.classificationModel.fit(X.values, y_desc[0].values)
        y_cont_pred = self.regressionModel.fit(X.values, y_cont[0].values)
        return self.join(y_desc_pred, y_cont_pred, self.getY(D))

    def join(self, y_desc_pred, y_cont_pred, y_cont):
        result = []
        for (dp, cp, c) in zip(y_desc_pred, y_cont_pred, y_cont[0].values):
            if dp == 0:
                result.append(c)
            else:
                result.append(cp)
        return result


    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

    def getY(self, D):
        pipelineYContinuous = Pipeline([
            ('NullToNaN', NullTransformer()),
            ('Continuous', ImputerTransformer('completed', 'mean'))
        ])
        return pipelineYContinuous.transform(D)

    def getYCont(self, D):
        pipelineYContinuous = Pipeline([
            ('NullToNaN', NullTransformer()),
            ('Continuous', YContinuousImputer())
        ])
        return pipelineYContinuous.transform(D)

    def getYDesc(self, D):
        pipelineY = Pipeline([
            ('NullToNaN', NullTransformer()),
            ('ClassY', ClassYTransformer())
        ])
        return pipelineY.transform(D)

    def getX(self, D):
        pipelineX = Pipeline([
            ('NullToNaN', NullTransformer()),
            ('featureType', FeatureUnion([
                ('revenue', RevenueTransformer()),
                ('units', UnitsTransformer()),
                ('datetime', DateTimeEncoderTransformer()),
                ('tsls', IntTransformer('tsls')),
                ('rating', IntTransformer('rating')),
                ('ttp', IntTransformer('ttp')),
                ('total_sessions', IntTransformer('total_sessions')),
                ('completed', FloatTransformer('completed')),
                ('win_rate', FloatTransformer('win_rate')),
                ('tries', IntTransformer('tries')),
                ('device', DeviceEncoderTransformer()),
                ('tbs', IntTransformer('tbs')),
                ('tsad', IntTransformer('tsad'))
            ])),
            ('ColumnNames1', ColumnNameTransformer(
                ['revenue', 'units', 'ls_date', 'tsls', 'rating', 'ttp', 'total_sessions', 'completed', 'win_rate',
                 'tries', 'device', 'tbs', 'tsad'])),
            ('featureImputer', FeatureUnion([
                ('ls_date_imputer', ImputerTransformer('ls_date', 'most_frequent')),
                ('tsls_imputer', ImputerTransformer('tsls', 'mean')),
                ('tbs_imputer', ImputerTransformer('tbs', 'mean')),
                ('completed_imputer', ImputerTransformer('completed', 'mean')),
                ('win_rate_imputer', ImputerTransformer('win_rate', 'mean')),
                ('tries_imputer', ImputerTransformer('tries', 'mean')),
                ('repeater',
                 RepeaterTransformer(['revenue', 'units', 'rating', 'ttp', 'total_sessions', 'device', 'tsad'])),
            ])),
            ('ColumnNames2', ColumnNameTransformer(
                ['ls_date', 'tsls', 'tbs', 'completed', 'win_rate', 'tries', 'revenue', 'units', 'rating', 'ttp',
                 'total_sessions', 'device', 'tsad']))
            ]
        )
        return pipelineX.transform(D)