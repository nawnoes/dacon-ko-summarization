import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
from util import jsonl_load
from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

class AbstrativeDataset(Dataset):
  def __init__(self,
               device,
               n_ctx = 1024,
               articles_max_length = 810,
               summary_max_length = 210,
               ):
    self.data =[]
    self.tokenizer = get_kogpt2_tokenizer()

    bos_token_id = [self.tokenizer.bos_token_id] # <s>
    eos_token_id = [self.tokenizer.eos_token_id] # </s>
    pad_token_id = [self.tokenizer.pad_token_id] # <pad>

    jsonl_datas = jsonl_load()
    # for dict_data in jsonl_datas:
    for dict_data in tqdm(jsonl_datas):
      articles = dict_data['article_original']
      abstractive_summary = dict_data['abstractive']

      tmp_str =''
      for article in articles:
        tmp_str += article

      # encode
      # truncate, if string exceed max length
      enc_tmp_str = self.tokenizer.encode(tmp_str, truncation= True, max_length=articles_max_length)
      enc_abstractive_summary = self.tokenizer.encode(abstractive_summary, truncation= True, max_length=summary_max_length)

      # <s> 요약할 문장 </s> 요약된 문장 </s>
      index_of_words = bos_token_id + enc_tmp_str+ eos_token_id + enc_abstractive_summary + eos_token_id
      pad_token_len = n_ctx - len(index_of_words)

      index_of_words += pad_token_id * pad_token_len

      self.data.append(torch.tensor(index_of_words).to(device))

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    item = self.data[index]
    return item

class ExtractiveDataset(Dataset):
  def __init__(self,
               data_path= './data/train.jsonl',
               num_label = 2, # 추출할것과 추출하지 않을 것들로
               device = 'cpu',
               max_seq_len = 512, # KoBERT max_length
               ):
    self.device = device
    self.data =[]
    self.tokenizer = get_tokenizer()

    cls_token_id = self.tokenizer.cls_token_id # [CLS]
    sep_token_id = self.tokenizer.sep_token_id # [SEP]
    pad_token_id = self.tokenizer.pad_token_id # [PAD]

    jsonl_datas = jsonl_load(data_path=data_path)
    # for dict_data in jsonl_datas:
    for dict_data in tqdm(jsonl_datas):
      articles = dict_data['article_original']
      extractive_indices = dict_data['extractive']

      index_of_words = None
      token_type_ids = None
      label= None
      token_num = None

      token_type_state = False

      for idx in range(len(articles)):
        label_state = True if idx in extractive_indices else False

        if idx == 0: # 맨 처음 문장인 경우
          index_of_words = [cls_token_id]
          token_type_ids = [int(token_type_state)]
          label = [int(label_state)]
          token_num = 1

        article = articles[idx]
        tmp_index = self.tokenizer.encode(article, add_special_tokens=False)
        num_tmp_index = len(tmp_index) + 1

        if token_num +  num_tmp_index <= max_seq_len:
          index_of_words += tmp_index + [sep_token_id]
          token_type_ids += [int(token_type_state)] * num_tmp_index

          label += [int(label_state)] * num_tmp_index
          token_num += num_tmp_index
          token_type_state = not token_type_state

        if token_num +  num_tmp_index > max_seq_len or idx == len(articles)-1 :
          # attention mask
          attention_mask = [1] * token_num

          # Padding Length
          padding_length = max_seq_len - token_num

          # Padding
          index_of_words += [pad_token_id] * padding_length # [PAD] padding
          token_type_ids += [token_type_state] * padding_length # last token_type_state padding
          attention_mask += [0] * padding_length # zero padding

          # Label Zero Padding
          label += [0] * padding_length

          # Data Append
          data = {
                  'input_ids': torch.tensor(index_of_words).to(self.device),
                  'token_type_ids': torch.tensor(token_type_ids).to(self.device),
                  'attention_mask': torch.tensor(attention_mask).to(self.device),
                  'labels': torch.tensor(label).to(self.device)
                 }
          self.data.append(data)

          # Data Initialization
          index_of_words = [cls_token_id]
          token_type_ids = [int(token_type_state)]
          label = [int(label_state)]
          token_num = 1
          token_type_state = False


  def __len__(self):
    return len(self.data)
  def __getitem__(self,index):
    item = self.data[index]
    return item

if __name__ == "__main__":
  dataset = AbstrativeDataset()
  # dataset2 = ExtractiveDataset()
  print(dataset)
  # print(dataset2)