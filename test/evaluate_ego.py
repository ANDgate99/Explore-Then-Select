import argparse
import json
import requests

def send_post_request(json_file):
    """
    Sends a POST request to the specified URL with the given JSON file.

    Parameters:
    - json_file (str): Path to the JSON file to be used in the request body.

    Returns:
    - Response object containing server's response.
    """

    url = "https://validation-server.onrender.com/api/upload/"
    headers = {
        "Content-Type": "application/json"
    }

    with open(json_file, 'r') as f:
        data = json.load(f)
    keys = list(data.keys())
    for key in keys:
        data[key] = ord(data[key]) - ord('A')
    response = requests.post(url, headers=headers, json=data)
    
    return response


parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str, default="/path/to/code/prediction/file")
parser.add_argument("--gt_file", type=str, default="/path/to/data/file")
parser.add_argument("--type", type=str, default="Subset", choices=["MC", "Subset"])
args = parser.parse_args()

if args.type == "MC":
    print("Evaluate the Total Dataset")
    response = send_post_request(args.file)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content:\n{response.text}")

elif args.type == "Subset":
    print("Evaluate Subset")
    total = 0
    acc = 0
    with open(args.gt_file, 'r') as file:
        ground_truth = json.load(file)
    with open(args.file, 'r') as file:
        result = json.load(file)
    for key in result:
        total += 1
        option = result[key]
        if ord(option) - ord('A') == ground_truth[key]:
            acc += 1
        # for i in range(len(option)):
        #     if ord(option[i]) - ord('A') == ground_truth[key]:
        #         acc += 1
        #         break
    print(f"Total: {total}, Correct: {acc}, Acc: {acc/total * 100}%")
    