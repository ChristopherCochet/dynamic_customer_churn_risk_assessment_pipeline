import training
import scoring
import ingestion
import deployment
import diagnostics
import reporting
import apicalls
import ast
import json
import os
import numpy as np
import os.path
from datetime import datetime

print("\n\n*** Full process - {} started!".format(datetime.now()))

######### read config files
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
logs_folder_path = config["logs_folder_path"]
prod_deployment_path = os.path.join(config["prod_deployment_path"])


print(
    "*** Full process - config files read ... \
    \n- input_folder_path:{} \n- output_folder_path:{} \n- logs_folder_path:{} \n- prod_deployment_path:{}".format(
        input_folder_path, output_folder_path, logs_folder_path, prod_deployment_path
    )
)

##################Check and read new data
# first, read
new_files_to_ingest = []
ingested_file = os.getcwd() + logs_folder_path + "ingestedfiles.txt"

# if files have never been ingested and the model never trained then run the full process end to end once
if not os.path.exists(ingested_file):
    print(
        "*** Full process - ingestion log file not found ... {}, running full process for the first time".format(
            ingested_file
        )
    )
    ingestion.merge_multiple_dataframe()
    training.train_model()
    scoring.score_model()
    deployment.store_model_into_pickle()
    diagnostics.model_predictions()
    diagnostics.dataframe_summary()
    diagnostics.dataframe_missing_data()
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()
    reporting.generate_confusion_matrix()
    exit(0)

# start full process logic if it has been run once
with open(ingested_file, "r") as file:
    ingested_file_list = ast.literal_eval(file.read())

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = os.listdir(os.getcwd() + input_folder_path)
for each_filename in filenames:
    if each_filename not in ingested_file_list:
        new_files_to_ingest.append(each_filename)
print(
    "*** Full process - checking files to ingest... \n- old files:{} \n- new files:{} ".format(
        ingested_file_list, new_files_to_ingest
    )
)

##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files_to_ingest) == 0:
    print("*** Full process - no new files found to ingest ... exiting !")
    exit(0)
else:
    print(
        "*** Full process - new files found to ingest ... {}".format(
            new_files_to_ingest
        )
    )
    ingestion.merge_multiple_dataframe()


##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
score_file = os.getcwd() + logs_folder_path + "latestscore.txt"
with open(score_file, "r") as file:
    old_score_list = ast.literal_eval(file.read())

print(
    "*** Full process - checking model score ... old score {} ".format(old_score_list)
)

# retain model on new data
print("*** Full process - re-training model ...")
training.train_model()

# score retrained model
print("*** Full process - scoring retrained model ...")
scoring.score_model()

# get new score
with open(score_file, "r") as file:
    latest_score = ast.literal_eval(file.read())

print(
    "*** Full process - checking scores ... \n- new score {} \n- old score {}".format(
        latest_score, old_score_list
    )
)
drift = latest_score < np.min(old_score_list)

##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if not drift:
    print("*** Full process - model has not drifted ... exiting !")
    exit(0)

##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
print("*** Full process - model drift found, re-running deployment... ")
deployment.store_model_into_pickle()

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
# check is API is deployed - if so run diagnostic on API, if not run diagnostics on local files
if apicalls.check_app_port():
    print("*** Full process - running diagnostics with new model on app ... ")
    apicalls.run_api_calls()
else:
    print("*** Full process - running diagnostics with new model without app ... ")
    diagnostics.model_predictions()
    diagnostics.dataframe_summary()
    diagnostics.dataframe_missing_data()
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()

print("*** Full process - running reporting with new model without app ... ")
reporting.generate_confusion_matrix()

print("\n\n*** Full process - completed!")
