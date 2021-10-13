import requests
import os
import json

with open("config.json", "r") as f:
    config = json.load(f)
logs_folder_path = config["logs_folder_path"]

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# http://localhost:8000/prediction?datafile=testdata/testdata.csv
# Call each API endpoint and store the responses
req_str = URL + "/prediction?datafile=testdata/testdata.csv"
print("API call 1 : {}".format(req_str))
response1 = requests.get(req_str).content

req_str = URL + "/scoring"
print("API call 2 : {}".format(req_str))
response2 = requests.get(req_str).content

req_str = URL + "/summarystats"
print("API call 3 : {}".format(req_str))
response3 = requests.get(req_str).content

req_str = URL + "/diagnostics"
print("API call 4 : {}".format(req_str))
response4 = requests.get(req_str).content

# combine all API responses
responses = b"\n\n".join([response1, response2, response3, response4])

# write the responses to your workspace
# write the combined outputs to a file call apireturns.txt.
log_file = os.getcwd() + logs_folder_path + "apireturns.txt"
with open(log_file, "wb") as file:
    file.write(responses)
