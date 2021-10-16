from flask import Flask, jsonify, request
import pickle
import diagnostics
import scoring
import json
import os

# Set up variables for use in our script
app = Flask(__name__)
# Fetch the SECRET in the .env file (not tracked using .gitignore)
app.secret_key = os.environ.get("SECRET_KEY")

# load config paths
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

# load model at app start-up
@app.before_first_request
def load_model():
    print("Loading model ...")
    # prediction_model = None
    model_name_file = "trainedmodel.pkl"
    with open(os.getcwd() + prod_deployment_path + model_name_file, "rb") as file:
        app.predictor = pickle.load(file)
    print("Flask app - load_model {} ...".format(app.predictor))


# Prediction Endpoint
@app.route("/prediction", methods=["GET", "OPTIONS"])
def predict():
    # take a dataset's file location as its input, and return predictions
    # call the prediction function you created in Step 3
    data_file = request.args.get("datafile")
    preds = diagnostics.model_predictions(app.predictor, data_file)

    return jsonify(preds)


# Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    # check the score of the deployed model
    # run the scoring.py
    score = scoring.score_model()

    return jsonify(score)  # add return value (a single F1 score number)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    # check means, medians, and modes for each column
    # run the summary statistics
    mystats = diagnostics.dataframe_summary()

    return jsonify(mystats)  # return a list of all calculated summary statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnose():
    # check timing and percent NA values
    # run the timing, missing data, and dependency check functions
    res = []
    timings = diagnostics.execution_time()
    res.append(timings)
    missing_val = diagnostics.dataframe_missing_data()
    res.append(missing_val)
    dependencies = diagnostics.outdated_packages_list()
    res.append(dependencies)

    return jsonify(res)  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
