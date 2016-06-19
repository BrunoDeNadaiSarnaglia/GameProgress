from pandas import DataFrame

__author__ = 'Bruno'

import csv

class TrainingData:

    def get(self):
        data = []
        file = open("./training_progress_predictor-3.csv", "rb")
        try:
            reader = csv.reader(file)
            iterator = iter(reader)
            columns = next(iterator)
            for row in iterator:
                data.append(row)
        finally:
            file.close()
        X = DataFrame(data)
        print columns
        X.columns = columns
        return X