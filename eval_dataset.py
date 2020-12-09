import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from util import jsonl_load
from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

class AbstrativeDataset(Dataset):
  def __init__(self,
               device,
               tokenizer,
               n_ctx = 1024,
               data_path='./data/extractive_test_v2.jsonl',
               articles_max_length = 810,
               summary_max_length = 210,
               ):
    self.data =[]
    self.tokenizer = tokenizer

    bos_token_id = [self.tokenizer.bos_token_id] # <s>
    eos_token_id = [self.tokenizer.eos_token_id] # </s>

    jsonl_datas = jsonl_load(data_path=data_path)
    for dict_data in tqdm(jsonl_datas):
      id = dict_data['id']
      articles = dict_data['article_original']

      tmp_str =''
      for article in articles:
        tmp_str += article

      # encode
      # truncate, if string exceed max length
      enc_tmp_str = self.tokenizer.encode(tmp_str, truncation= True, max_length=articles_max_length)

      # <s> 요약할 문장 </s> 요약된 문장 </s>
      index_of_words = bos_token_id + enc_tmp_str+ eos_token_id

      self.data.append({
        'id':id,
        'input':torch.tensor([index_of_words]).to(device)
      })

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    item = self.data[index]
    return item

class ExtractiveDataset(Dataset):
  def __init__(self,
               tokenizer,
               device = 'cpu',
               data_path='./data/extractive_test_v2.jsonl',
               max_seq_len = 512, # KoBERT max_length
               ):
    self.device = device
    self.data =[]
    self.tokenizer = tokenizer

    cls_token_id = self.tokenizer.cls_token_id # [CLS]
    sep_token_id = self.tokenizer.sep_token_id # [SEP]
    pad_token_id = self.tokenizer.pad_token_id # [PAD]

    jsonl_datas = jsonl_load(data_path = data_path)
    for dict_data in tqdm(jsonl_datas):
      id = dict_data["id"]
      articles = dict_data['article_original']

      index_of_words = None
      token_type_ids = None
      token_num = None

      token_type_state = False

      for idx in range(len(articles)):

        if idx == 0: # 맨 처음 문장인 경우
          index_of_words = [cls_token_id]
          token_type_ids = [int(token_type_state)]
          token_num = 1

        article = articles[idx]
        tmp_index = self.tokenizer.encode(article, add_special_tokens=False)
        num_tmp_index = len(tmp_index) + 1

        if token_num +  num_tmp_index <= max_seq_len:
          index_of_words += tmp_index + [sep_token_id]
          token_type_ids += [int(token_type_state)] * num_tmp_index

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

          # Data Append
          # Need additional 1 dimension for inference
          input = {
                  'input_ids': torch.tensor([index_of_words]).to(self.device),
                  'token_type_ids': torch.tensor([token_type_ids]).to(self.device),
                  'attention_mask': torch.tensor([attention_mask]).to(self.device),
                 }
          self.data.append({
            "id":id,
            "input":input
          })

          # Data Initialization
          index_of_words = [cls_token_id]
          token_type_ids = [int(token_type_state)]
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