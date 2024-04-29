import os
import sys
import time
import glob
import pickle
import shutil
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import polars as pl
from pathlib import Path

MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

class TextLoader(Dataset):
    def __init__(self, file, tokenizer):
        df = pl.read_parquet(file)
        print('File name', file)
        print('Number of records in file', len(df))
        self.file = tokenizer(list(df['reddit_text']), padding=True, truncation=True, max_length=64, return_tensors='pt')   
        self.file = self.file['input_ids']

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        return self.file[idx]

class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def forward(self, input):
        return self.model(input)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = SentimentModel()
device_staging = 'cuda:0' if torch.cuda.device_count() > 1 else 'cuda'
model = model.to(device_staging)

def get_all_files(path):
    return list(Path(path).glob("split_*.parquet"))

def interpret_logits(npy_file):
    logits = np.load(npy_file, allow_pickle=True)
    sentiments = [np.argmax(logit) if np.argmax(logit) == 0 else (-1 if np.argmax(logit) == 1 else 0) for batch in logits for logit in batch]
    return sentiments

def update_dataframe_with_sentiments(base_dir, parquet_path, npy_file_path):
    df = pl.read_parquet(parquet_path)
    sentiments = interpret_logits(npy_file_path)
    df['text_sentiment'] = sentiments
    df.to_parquet(parquet_path)

def combine_parquet_files(base_dir, output_file):
    search_pattern = os.path.join(base_dir, "split_*.parquet")
    parquet_files = glob.glob(search_pattern)
    table_list = [pq.read_table(parquet_path) for parquet_path in parquet_files if parquet_files]
    if table_list:
        combined_table = pa.concat_tables(table_list)
        pq.write_table(combined_table, output_file, compression='zstd')
    else:
        print("No Parquet files found in the directory.")

def add_summed_sentiments(df):
    replies_agg = df.groupby("reddit_parent_id").agg([
        pl.sum("text_sentiment").alias("summed_sentiments"),
        (pl.sum("text_sentiment").abs()).alias("absolute_summed_sentiment")
    ]).rename({"reddit_parent_id": "reddit_name"})
    result_df = df.join(replies_agg, on="reddit_name", how="left")
    result_df = result_df.with_columns([
        pl.col("summed_sentiments").fill_null(0).cast(int),
        pl.col("absolute_summed_sentiment").fill_null(0).cast(pl.Int32)
    ])
    return result_df
