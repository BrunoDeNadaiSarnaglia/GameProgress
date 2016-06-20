import numpy
from pandas import DataFrame
from sklearn.linear_model import Lasso
from sklearn.pipeline import FeatureUnion
from sklearn import linear_model, decomposition, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from Data.LoadTestData import TestData
from FinalModel import FinalModel
from Fitter.NaiveBayesianFitter import NaiveBayesianFitter
from Fitter.NearestNeighborsFitter import NearestNeighborsFitter
from Fitter.RandomForestFitter import RandomForestFitter
from Fitter.RidgeRegression import RidgeRegressionFitter
from Fitter.SVMFitter import SVMFitter
from Fitter.SVMFitterBestParameters import SVMFitterBestParameters

from Pipeline.ClassYTransformer import ClassYTransformer
from Pipeline.ImputerTransformer import ImputerTransformer
from Pipeline.ColumnNameTransformer import ColumnNameTransformer
from Pipeline.DateTimeEncoderTransformer import DateTimeEncoderTransformer
from Pipeline.DeviceEncoderTransformer import DeviceEncoderTransformer
from Pipeline.FloatTransformer import FloatTransformer
from Pipeline.IntTransformer import IntTransformer
from Pipeline.NullTransformer import NullTransformer
from Fitter.LinearRegressionFitter import LinearRegressionFitter
from Pipeline.RepeaterTransformer import RepeaterTransformer
from Pipeline.RevenueTransformer import RevenueTransformer
from Pipeline.UnitsTransformer import UnitsTransformer
from Data.LoadTrainingData import TrainingData
from Pipeline.YContinuousImputer import YContinuousImputer
from Regression.BayesianRidgeRegression import BayesianRidgeRegression
from Regression.ElasticNetRegression import ElasticNetRegression
from Regression.LassoRegression import LassoRegression
from Regression.LassoRegressionBestParameters import LassoRegressionBestParameter
from Regression.PassiveAggressiveRegressor import PassiveAggressiveRegression
from Regression.SVRRegression import SVRRegression




def main():
    file = TrainingData()
    data = file.get()
    DTraining = DataFrame(data)
    file = TestData()
    data = file.get()
    DTest = DataFrame(data)

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

    X = pipelineX.transform(DTraining)
    y = pipelineY.transform(DTraining)
    pipelineSVC = Pipeline([
        ('pca', decomposition.PCA(n_components=8)),
        ('svc', SVC(kernel="rbf", max_iter=-1, C=0.1))
    ])

    svc =SVC(kernel="rbf", max_iter=-1, C=0.1)
    y_pred = svc.fit(X.values, y[0].values)

    pipelineYContinuous = Pipeline([
        ('NullToNaN', NullTransformer()),
        ('Continuous', YContinuousImputer())
    ])

    y_cont = pipelineYContinuous.transform(DTraining)
    pipelineLasso = Pipeline([
        ('pca', decomposition.PCA(n_components=4)),
        ('lasso', Lasso(alpha=0.1))
    ])

    lasso = Lasso(alpha=0.1)
    y_cont_pred = lasso.fit(X.values, y_cont[0].values)



    pipelineYContinuous2 = Pipeline([
        ('NullToNaN', NullTransformer()),
        ('completed', ImputerTransformer('completed', 'mean'))
    ])

    X_test = pipelineX.transform(DTest)
    # X_test = preprocessing.scale(X_test)
    y_test_cont = pipelineYContinuous2.transform(DTest)
    y_test_pred_desc = svc.predict(X_test)
    y_test_cont_pred = lasso.predict(X_test)

    r = ""
    for (a, b, c) in zip(y_test_cont, y_test_pred_desc, y_test_cont_pred):
        r += str(a[0]) + "," + str(b) + "," + str(c) + "\n"
    with open("Output.csv", "w") as text_file:
        text_file.write(r)

main()

