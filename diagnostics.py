import pandas as pd
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_folder_path = config["output_folder_path"]
logs_folder_path = config["logs_folder_path"]

##################Function to get model predictions
def model_predictions(datafile=None):
    # read the deployed model and a test dataset, calculate predictions

    features_var = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target_var = "exited"

    # load model
    model_name_file = "trainedmodel.pkl"
    with open(os.getcwd() + prod_deployment_path + model_name_file, "rb") as file:
        model = pickle.load(file)
    print("Diagnostics - model_predictions load model {} ...".format(model))

    # read in trainig data
    if datafile is None:
        datafile = os.getcwd() + test_data_path + "testdata.csv"
        test_df = pd.read_csv(datafile)
    else:
        test_df = pd.read_csv(datafile)
    print("Diagnostics - model_predictions load test file {} ...".format(datafile))

    # define X & Y
    X = test_df[features_var]
    y = test_df[target_var]

    # retrieve predictions
    predicted = model.predict(X)
    print(
        "Diagnostics - model_predictions generating {} predictions ...".format(
            len(predicted)
        )
    )
    # return value should be a list containing all predictions

    return predicted.tolist()


##################Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    features_var = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]

    # read in the data file
    data_file = os.getcwd() + output_folder_path + "finaldata.csv"
    data_df = pd.read_csv(data_file)
    print("Diagnostics - dataframe_summary ...")

    # summary statistics here - means, medians, and standard deviation
    statistics_list = data_df[features_var].mean().to_list()
    statistics_list.append(data_df[features_var].median().to_list())
    statistics_list.append(data_df[features_var].std().to_list())

    return [
        statistics_list
    ]  # return value should be a list containing all summary statistics


##################Function to get missing data
def dataframe_missing_data():
    # Count the number of NA values in each column of your dataset
    # Then, calculate what percent of each column consists of NA values

    print("Diagnostics - dataframe_summary ...")

    # read in the data file
    data_file = os.getcwd() + output_folder_path + "finaldata.csv"
    data_df = pd.read_csv(data_file)
    print("Diagnostics - dataframe_missing_data ...")

    # compute % missing values
    res = data_df.isnull().mean() * 100
    return res.to_list()


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    print("Diagnostics - execution_time ...")

    starttime = timeit.default_timer()
    os.system("python3 ingestion.py")
    timing_ingestion = timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system("python3 training.py")
    timing_training = timeit.default_timer() - starttime

    return [
        timing_ingestion,
        timing_training,
    ]  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    # get a list of outdated dependencies
    outdated = subprocess.check_output(["pip", "list", "--outdated"])
    print("Diagnostics - outdated_packages_list {}...".format(outdated.decode("utf-8")))

    # track outdates packages file
    log_file = os.getcwd() + logs_folder_path + "outdated.txt"
    with open(log_file, "wb") as file:
        file.write(outdated)

    return outdated.decode("utf-8")


if __name__ == "__main__":
    res = model_predictions()
    print(res)

    res = dataframe_summary()
    print(res)

    res = dataframe_missing_data()
    print(res)

    res = execution_time()
    print(res)

    outdated_packages_list()
