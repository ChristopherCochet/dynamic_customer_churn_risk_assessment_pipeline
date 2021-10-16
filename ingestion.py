import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
logs_folder_path = config["logs_folder_path"]

#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    directories = [input_folder_path]
    final_dataframe = None
    print("Ingestion - merge_multiple_dataframe for directories {}".format(directories))

    # create a single dataframe from files read
    for directory in directories:
        filenames = os.listdir(os.getcwd() + directory)
        print("Ingestion - merge_multiple_dataframe merging files {}".format(filenames))

        for each_filename in filenames:
            if final_dataframe is None:
                currentdf = pd.read_csv(os.getcwd() + directory + each_filename)
                final_dataframe = currentdf
            else:
                final_dataframe = final_dataframe.append(currentdf).reset_index(
                    drop=True
                )

    # write dataframe to output csv file and directory
    output_path = os.getcwd() + output_folder_path
    output_file = output_path + "finaldata.csv"
    print(
        "Ingestion - merge_multiple_dataframe writing de-duplicated output to {}".format(
            output_file
        )
    )
    final_dataframe.drop_duplicates(keep="first").to_csv(output_file, index=False)

    # track files read and saved
    log_file = os.getcwd() + logs_folder_path + "ingestedfiles.txt"
    log_str = str(filenames)
    with open(log_file, "w") as file:
        file.write(log_str)


if __name__ == "__main__":
    merge_multiple_dataframe()
