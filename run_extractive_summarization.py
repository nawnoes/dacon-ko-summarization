"""
submission.csv
   id        summary
[아이디명]     [추출 요약문]
"""
import csv
import torch
from tqdm import tqdm

from kobert_transformers import get_tokenizer
from eval_dataset import ExtractiveDataset
from model.kobert import KoBERTforExtractiveSummarization

"""
id, article_original

"""
# config
tokenizer = get_tokenizer()
dir_path="."
ckpt_path = f'{dir_path}/checkpoint/kobert-extractive.pth'
csv_path = f'{dir_path}/data/extractive_summary_kobert.csv'
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# load data
eval_datas = ExtractiveDataset(tokenizer=tokenizer,data_path='./data/eval_test.jsonl')

# load kobert checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)
model = KoBERTforExtractiveSummarization()
model.load_state_dict(checkpoint['model_state_dict'])

# eval mode
model.eval()

# eval csv_path
# 한글 깨짐 방지 설정 encoding='utf-8-sig'
f = open(csv_path, 'w', encoding='utf-8-sig', newline='')
wr = csv.writer(f)
wr.writerow(['id','summary'])

result_data = {}
for data in tqdm(eval_datas):
  id = data['id']
  input = data['input']

  output = model(**input)
  logit = output['logits'][0]
  softmax_logit = torch.softmax(logit, dim=1)
  argmax = torch.argmax(softmax_logit, dim=1)


  extractive_index = torch.nonzero(argmax.bool()).view(-1)
  extractive_input_ids = torch.index_select(input=input['input_ids'],index=extractive_index,dim=-1)
  result = tokenizer.decode(extractive_input_ids.squeeze(),skip_special_tokens=True)
  print(result)

  if id in result_data.keys():
    result_data[id] += result
  else:
    result_data[id] = result

for key in result_data:
  wr.writerow([key,result_data[key]])

f.close()

