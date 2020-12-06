# python 경고 메세지 안뜨
import warnings
warnings.filterwarnings(action='ignore')

import json

from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

def jsonl_load(data_path = './data/train.jsonl'):
  data = []
  with open(data_path, 'r') as json_file:
      json_list = list(json_file)

  for json_str in json_list:
      data.append(json.loads(json_str))

  return data

def test(data_path = './data/train.jsonl'):
  data = []
  with open(data_path, 'r') as json_file:
      json_list = list(json_file)

  sum_len = 0
  count = 0
  for json_str in json_list:
      json_data = json.loads(json_str)
      print(len(json_data['article_original']))
      sum_len += len(json_data['article_original'])
      count += 1
  print('average article_original len - ', sum_len/count)
  return data

def token_num(data_path = './data/train.jsonl'):
  data = []
  with open(data_path, 'r') as json_file:
      json_list = list(json_file)

  bert_tok = get_tokenizer()
  gpt_tok = get_kogpt2_tokenizer()

  bert_tok_num = 0
  gpt_tok_num = 0

  count = 0
  for json_str in json_list:
      json_data = json.loads(json_str)
      tmp_str = json_data['abstractive']
      # for arti_str in json_data['article_original']:
      #   tmp_str += arti_str
      bert_tok_num = max(bert_tok_num, len(bert_tok.encode(tmp_str,max_length=512, truncation=True)))
      gpt_tok_num = max(gpt_tok_num, len(gpt_tok.encode(tmp_str, max_length=512, truncation=True)))

      # print(len(json_data['article_original']))
      # sum_len += len(json_data['article_original'])
      # count += 1
  # print('average article_original len - ', sum_len/count)
  print('max bert token len:', bert_tok_num)
  print('max gpt token len:', gpt_tok_num)

  # return data

if __name__ == '__main__':
  # data= jsonl_load()
  # print(data)

  token_num()