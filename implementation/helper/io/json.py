import json
import os


def json_file_reader(filePath):
    json_string = None
    with open(filePath, 'r') as file:
        json_string = file.read()
    return json.loads(json_string)


def dict_to_json(filePath, data):
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, 'w') as outfile:
        json.dump(data, outfile)
