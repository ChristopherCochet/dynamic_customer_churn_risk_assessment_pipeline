import requests
import os
import json
import socket

with open("config.json", "r") as f:
    config = json.load(f)
logs_folder_path = config["logs_folder_path"]
output_folder_path = config["output_folder_path"]
model_path = os.path.join(config["output_model_path"])

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


def check_app_port(port=8000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 8000))
    if result == 0:
        print("API calls - check_app_port() ... port {} is open".format(port))
    else:
        print("API calls - check_app_port() ... port {} is not open".format(port))
    sock.close()
    return result == 0


def run_api_calls():

    # http://localhost:8000/prediction?datafile=testdata/testdata.csv
    # Call each API endpoint and store the responses
    req_str = URL + "/prediction?datafile=testdata/testdata.csv"
    print("API calls - API call 1 : {}".format(req_str))
    response1 = requests.get(req_str).content

    req_str = URL + "/scoring"
    print("API calls - API call 2 : {}".format(req_str))
    response2 = requests.get(req_str).content

    req_str = URL + "/summarystats"
    print("API calls - API call 3 : {}".format(req_str))
    response3 = requests.get(req_str).content

    req_str = URL + "/diagnostics"
    print("API calls - API call 4 : {}".format(req_str))
    response4 = requests.get(req_str).content

    # combine all API responses
    responses = b"\n\n".join([response1, response2, response3, response4])

    # write the responses to your workspace
    # write the combined outputs to a file call apireturns.txt.

    if model_path == "/models/":
        log_file = os.getcwd() + model_path + "apireturns2.txt"
    else:
        log_file = os.getcwd() + logs_folder_path + "apireturns.txt"

    print("API calls - writing app outputs to {}".format(log_file))
    with open(log_file, "wb") as file:
        file.write(responses)


if __name__ == "__main__":

    if check_app_port():
        run_api_calls()
