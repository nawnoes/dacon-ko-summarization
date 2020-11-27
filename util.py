import json


def jsonl_load(data_path = 'data/train.jsonl'):
  data = []
  with open(data_path, 'r') as json_file:
      json_list = list(json_file)

  for json_str in json_list:
      data.append(json.loads(json_str))

  return data

if __name__ == '__main__':
  data= jsonl_load()
  print(data)