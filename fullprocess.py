import training
import scoring
import ingestion
import deployment
import diagnostics
import reporting
import ast
import json
import os
import np

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
logs_folder_path = config["logs_folder_path"]
prod_deployment_path = os.path.join(config["prod_deployment_path"])

##################Check and read new data
# first, read
ingested_file = os.getcwd() + logs_folder_path + "ingestedfiles.txt"
with open(ingested_file, "r") as file:
    ingested_file_list = ast.literal_eval(file.read())

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files_to_ingest = []
filenames = os.listdir(os.getcwd() + input_folder_path)
for each_filename in filenames:
    if each_filename not in ingested_file_list:
        new_files_to_ingest.append(each_filename)

##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files_to_ingest) == 0:
    print("Full process - no new files to ingest ... exiting !")
    exit(0)
else:
    print("Full process - new files to ingest ... {}".format(new_files_to_ingest))
    ingestion.merge_multiple_dataframe()


##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
score_file = os.getcwd() + logs_folder_path + "latestscore.txt"
with open(score_file, "r") as file:
    old_score_list = ast.literal_eval(file.read())

# retain model on new data
print("Full process - re-training model ...")
training.train_model()

# score retrained model
print("Full process - scoring retrained model ...")
scoring.score_model()

# get new score
with open(score_file, "r") as file:
    latest_score = ast.literal_eval(file.read())

print(
    "Full process - new score {} vs. old score {}".format(latest_score, old_score_list)
)
drift = latest_score < np.min(old_score_list)
##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if not drift:
    print("Full process - model has not drifted ... exiting !")
    exit(0)

##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
print("Full process - model drift found, re-running deployment... ")
deployment.store_model_into_pickle()

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
print("Full process - running diagnostics with new model... ")
reporting.generate_confusion_matrix()
