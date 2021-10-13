from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])
logs_folder_path = config["logs_folder_path"]

#################Function for training the model
def train_model():

    print("Training - train_model")
    # define X & Y
    features_var = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target_var = "exited"

    # read in trainig data
    train_file = os.getcwd() + dataset_csv_path + "finaldata.csv"
    training_df = pd.read_csv(train_file)
    print("Training - train_model load training df {} ...".format(training_df.shape))

    # train_test_split()

    X = training_df[features_var]
    y = training_df[target_var]

    # use this logistic regression for training
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="lbfgs",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    print("Training - train_model fitting model {} ...".format(lr))

    # fit the logistic regression to your data
    model = lr.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_name_file = "trainedmodel.pkl"
    pickle.dump(model, open(os.getcwd() + model_path + model_name_file, "wb"))
    print(
        "Training - train_model saving model to {} ...".format(
            os.getcwd() + model_path + model_name_file
        )
    )

    # track saved model file
    log_file = os.getcwd() + logs_folder_path + "model.txt"
    log_str = (
        "Training - train_model() - model : "
        + os.getcwd()
        + model_path
        + model_name_file
    )
    with open(log_file, "w") as file:
        file.write(log_str)


if __name__ == "__main__":
    train_model()
