import pickle
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
logs_folder_path = config["logs_folder_path"]
model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])

##############Function for reporting
def generate_confusion_matrix():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    features_var = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target_var = "exited"

    # load model
    model_name_file = "trainedmodel.pkl"
    with open(os.getcwd() + model_path + model_name_file, "rb") as file:
        model = pickle.load(file)
    print("Reporting - generate_confusion_matrix load model {} ...".format(model))

    # read in trainig data
    test_file = os.getcwd() + test_data_path + "testdata.csv"
    test_df = pd.read_csv(test_file)
    print(
        "Reporting - generate_confusion_matrix load test file {} ...".format(test_file)
    )

    # define X & Y
    X = test_df[features_var]
    y = test_df[target_var]

    # plot and save confusion matrix
    if model_path == "/models/":
        cm_file = os.getcwd() + model_path + "confusionmatrix2.png"
    else:
        cm_file = os.getcwd() + model_path + "confusionmatrix.png"
    print(
        "Reporting - generate_confusion_matrix saving confusion matrix to {}".format(
            cm_file
        )
    )
    plot_confusion_matrix(model, X, y)
    plt.title("Logistic Regression - Confusion matrix")
    plt.savefig(cm_file)


if __name__ == "__main__":
    generate_confusion_matrix()
