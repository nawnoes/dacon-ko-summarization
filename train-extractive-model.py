import json

data_path = 'data/train.jsonl'

with open(data_path, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    print("result: {}".format(result))
    print(isinstance(result, dict))
