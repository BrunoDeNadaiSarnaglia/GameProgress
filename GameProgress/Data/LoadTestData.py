__author__ = 'Bruno'
from pandas import DataFrame

__author__ = 'Bruno'

import csv

class TestData:

    def get(self):
        data = []
        file = open("./test_progress_predictor-4.csv", "rb")
        try:
            reader = csv.reader(file)
            iterator = iter(reader)
            columns = next(iterator)
            for row in iterator:
                data.append(row)
        finally:
            file.close()
        X = DataFrame(data)
        X.columns = columns
        return X