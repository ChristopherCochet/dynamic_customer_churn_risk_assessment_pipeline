import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import json
from sklearn import metrics
from sklearn.metrics import f1_score


#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])
logs_folder_path = config["logs_folder_path"]

#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    features_var = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target_var = "exited"

    # load model
    model_name_file = "trainedmodel.pkl"
    with open(os.getcwd() + model_path + model_name_file, "rb") as file:
        model = pickle.load(file)
    print("Scoring - score_model load model {} ...".format(model))

    # read in trainig data
    test_file = os.getcwd() + test_data_path + "testdata.csv"
    test_df = pd.read_csv(test_file)
    print("Scoring - score_model load test file {} ...".format(test_file))

    # define X & Y
    X = test_df[features_var]
    y = test_df[target_var]

    # retrive predictions and compute f1 score
    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)
    print("Scoring - score_model f1 score {} ...".format(f1score))

    # Write the F1 score to a file in your workspace called latestscore.txt.
    # track saved model file
    log_file = os.getcwd() + logs_folder_path + "latestscore.txt"
    log_str = str(f1score)
    with open(log_file, "w") as file:
        file.write(log_str)

    return f1score


if __name__ == "__main__":
    score_model()

