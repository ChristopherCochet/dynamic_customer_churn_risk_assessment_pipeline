import os
import json
import shutil

# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config["prod_deployment_path"])
logs_folder_path = config["logs_folder_path"]
model_path = os.path.join(config["output_model_path"])

# function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and
    # the ingestfiles.txt file into the deployment directory
    print("Deployment - store_model_into_pickle ...")
    target_path = os.getcwd() + prod_deployment_path

    # Copy files to production_deployment directory
    f1 = os.getcwd() + logs_folder_path + "ingestedfiles.txt"
    f2 = os.getcwd() + logs_folder_path + "latestscore.txt"
    files_to_copy = [f1, f2]
    for f in files_to_copy:
        print(f"Deployment - store_model_into_pickle copying {f} to \n  {target_path}")
        shutil.copy2(f, target_path)

    f3 = os.getcwd() + model_path + "trainedmodel.pkl"
    files_to_copy = [f3]
    for f in files_to_copy:
        print(f"Deployment - store_model_into_pickle copying {f} to \n  {target_path}")
        shutil.copy2(f, target_path)


if __name__ == "__main__":
    store_model_into_pickle()
