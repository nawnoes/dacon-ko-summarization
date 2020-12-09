"""
submission.csv
   id        summary
[아이디명]     [추출 요약문]

import csv
f = open('output.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([1, "김정수", False])
wr.writerow([2, "박상미", True])
f.close()

"""

"""
submission.csv
   id        summary
[아이디명]     [추출 요약문]
"""
import csv
import torch
from tqdm import tqdm

from kogpt2_transformers import get_kogpt2_tokenizer
from eval_dataset import AbstrativeDataset
from model.kogpt2 import AbstractiveKoGPT2

"""
id, article_original

"""
# config
tokenizer = get_kogpt2_tokenizer()
dir_path="."
ckpt_path = f'{dir_path}/checkpoint/kogpt2-abstractive-10-epoch.pth'
csv_path = f'{dir_path}/data/abstractive_summary.csv'
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# load data
eval_datas = AbstrativeDataset(tokenizer=tokenizer,device=device,data_path='./data/eval_test.jsonl')

# load kobert checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)
model = AbstractiveKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])

# eval mode
model.eval()

# eval csv_path
# 한글 깨짐 방지 설정 encoding='utf-8-sig'
f = open(csv_path, 'w', encoding='utf-8-sig', newline='')
wr = csv.writer(f)
wr.writerow(['id','summary'])

for data in tqdm(eval_datas):
# for data in eval_datas:
  id = data['id']
  input_ids = data['input']

  sample_output = model.generate(input_ids=input_ids,max_length=1024)
  summary = tokenizer.decode(sample_output[0].tolist()[len(input_ids[0]):-1])
  wr.writerow([id, summary.replace('</s>','')])

f.close()

