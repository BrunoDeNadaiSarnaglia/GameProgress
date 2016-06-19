import numpy
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from Pipeline.ClassYTransformer import ClassYTransformer
from Pipeline.ImputerTransformer import ImputerTransformer
from Pipeline.ColumnNameTransformer import ColumnNameTransformer
from Pipeline.DateTimeEncoderTransformer import DateTimeEncoderTransformer
from Pipeline.DeviceEncoderTransformer import DeviceEncoderTransformer
from Pipeline.FloatTransformer import FloatTransformer
from Pipeline.IntTransformer import IntTransformer
from Pipeline.NullTransformer import NullTransformer
from Pipeline.RepeaterTransformer import RepeaterTransformer
from Pipeline.RevenueTransformer import RevenueTransformer
from Pipeline.UnitsTransformer import UnitsTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from Data.LoadTrainingData import TrainingData
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score

def main():
    file = TrainingData()
    data = file.get()
    D = DataFrame(data)
    pipelineX = Pipeline(
        [
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
    pipelineY = Pipeline([
        ('NullToNaN', NullTransformer()),
        ('ClassY', ClassYTransformer())
    ])
    pipelineFit = Pipeline([
        ('pca', decomposition.PCA()),
        ('logistic', linear_model.LogisticRegression())
    ])
    X = pipelineX.transform(D)
    y = pipelineY.transform(D)
    n_components = [2, 4, 6, 8, 10, 12]
    Cs = numpy.logspace(-4, 4, 3)

    estimator = GridSearchCV(pipelineFit,
                             dict(pca__n_components=n_components, logistic__C=Cs))
    estimator.fit(X.values, y[0].values)
    print estimator.best_estimator_.score(X, y)
    y_pred = estimator.best_estimator_.predict(X)
    print precision_score(y[0].values, y_pred)


main()

