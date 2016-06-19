__author__ = 'Bruno'

import csv

class TrainingData:

    def get(self):
        data = []
        file = open("./training_progress_predictor-3.csv", "rb")
        try:
            reader = csv.reader(file)
            iterator = iter(reader)
            next(iterator)
            for row in iterator:
                data.append(row)
        finally:
            file.close()
        return data