__author__ = 'Bruno'

import csv

class TrainingData:


    def get(self):
        file = open("./training_progress_predictor-3.csv", "rb")
        try:
            reader = csv.reader(file)
            for row in reader:
                print row
        finally:
            file.close()