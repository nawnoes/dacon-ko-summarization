# dacon-ko-abstract-extract
데이콘 한국어 문서 추출 및 생성요약 AI 경진대회 정리
## 대회 목표
다양한 주제의 한국어 원문으로부터 추출요약문과 생성요약문을 도출해낼 수 있도록 인공지능을 개발

## 데이터 
### jsonl
jsonl 파일은 Json Line 형식의 파일로, 한줄에 json 객체 하나로 되어 있는 파일 형식을 말한다. 

#### 학습 데이터
###### train.jsonl
- media : 기사 미디어
- id : 각 데이터 고유 번호
- article_original : 전체 기사 내용, 문장별로 split. 
    + 최대 길이는 97
    + 평균 길이는 13
    + 최대 토큰 길이
        * kobert tokenizer max length: 1250
        * kogpt tokenizer max length: 1029  
- abstractive : 사람이 생성한 요약문. 최대 토큰 길이 213. koGPT2 기준
- extractive : 사람이 추출한 요약문 3개의 index
#### 테스트 데이터
###### abstractive_test_v2.jsonl
###### extractive_test_v2.jsonl

#### 제출 데이터
xxxx_test_v2.json의 추론결과를 csv 파일로 출

## 생성요약 (Abstrative)
KoGPT2의 max_len = 1024
## 추출요약 (Extrative)
KoBERT의 max_len = 512