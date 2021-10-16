import subprocess
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)

logs_folder_path = config["logs_folder_path"]
cron_file = os.getcwd() + logs_folder_path + "cron.txt"

print(cron_file)

with open(cron_file, "a") as output:
    subprocess.call(
        [
            "/home/cmc265/miniconda/envs/churn_risk/bin/python",
            "/home/cmc265/udacity/mlops-nanodegree/dynamic_customer_churn_risk_assessment_pipeline/fullprocess.py",
        ],
        stdout=output,
    )

