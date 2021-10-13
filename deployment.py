from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config["prod_deployment_path"])
logs_folder_path = config["logs_folder_path"]
model_path = os.path.join(config["output_model_path"])

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    print("Deployment - store_model_into_pickle ...")

    # Copy files to production_deployment directory
    cmd = (
        "cp "
        + os.getcwd()
        + logs_folder_path
        + "* "
        + os.getcwd()
        + prod_deployment_path
    )
    print(cmd)
    os.system(cmd)

    cmd = "cp " + os.getcwd() + model_path + "* " + os.getcwd() + prod_deployment_path
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    store_model_into_pickle()
