from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
import scoring
import json
import os


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
# prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=["GET", "OPTIONS"])
def predict():
    # take a dataset's file location as its input, and return the outputs of the prediction
    # call the prediction function you created in Step 3
    data_file = request.args.get("datafile")
    preds = diagnostics.model_predictions(data_file)

    return str(preds)


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    # check the score of the deployed model
    # run the scoring.py
    score = scoring.score_model()

    return str(score)  # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    # check means, medians, and modes for each column
    # run the summary statistics
    mystats = diagnostics.dataframe_summary()

    return str(mystats)  # return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnose():
    # check timing and percent NA values
    # run the timing, missing data, and dependency check functions
    res = []
    timigs = diagnostics.execution_time()
    res.append(timigs)
    missing_val = diagnostics.dataframe_missing_data()
    res.append(missing_val)
    dependencies = diagnostics.outdated_packages_list()
    res.append(dependencies)

    return str(res)  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)

