import pandas as pd
import os
import glob
import json

# Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
logs_folder_path = config["logs_folder_path"]

# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    input_folder = os.getcwd() + input_folder_path
    final_dataframe = None
    print(
        "Ingestion - merge_multiple_dataframe for directory \n  {}".format(input_folder)
    )

    # create a single dataframe from files read
    filenames = glob.glob(f"{input_folder}/*.csv")
    print(
        "Ingestion - merge_multiple_dataframe ingestings files \n  {}".format(filenames)
    )
    final_dataframe = pd.concat(map(pd.read_csv, filenames))

    # write the dedup dataframe to output csv file and directory
    output_path = os.getcwd() + output_folder_path
    output_file = output_path + "finaldata.csv"
    print(
        "Ingestion - merge_multiple_dataframe writing de-duplicated output to \n  {}".format(
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
