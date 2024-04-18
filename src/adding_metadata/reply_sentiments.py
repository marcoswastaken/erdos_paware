import polars as pl
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from datasets import Dataset

# Ensure the GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForTokenClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis").to(device)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

def batch_sentiment(texts):
    """Process sentiment analysis in batches using datasets to utilize GPU more efficiently."""
    if not texts:
        return ['null'] * len(texts)  # Handle empty inputs
    text_dataset = Dataset.from_dict({'text': texts})

    # Make sure the process_sentiment handles batches
    def process_sentiment(batch):
        # The `batch['text']` is a list of texts
        processed_texts = nlp(batch['text'])
        return {'sentiment': [res[0]['entity'] if res else 'null' for res in processed_texts]}

    # Ensure batched=True and set a reasonable batch size
    sentiment_dataset = text_dataset.map(process_sentiment, batched=True, batch_size=32)
    return [item['sentiment'] for item in sentiment_dataset]

def fetch_texts_and_apply_sentiment(reply_ids, df):
    """Fetch texts for given reply IDs and apply sentiment analysis in batch."""
    reply_texts = df.filter(pl.col("reddit_name").is_in(reply_ids))["reddit_text"].to_list()
    return batch_sentiment(reply_texts) if reply_texts else ['null'] * len(reply_ids)

def add_reply_sentiments(df: pl.DataFrame) -> pl.DataFrame:
    """Add a column called 'reply_sentiment' containing sentiments for each set of reply_ids."""
    sentiments = [fetch_texts_and_apply_sentiment(ids, df) if ids else ['null'] for ids in df["reply_ids"].to_list()]
    return df.with_columns(pl.Series("reply_sentiments", sentiments))

def add_text_sentiment(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a column called 'text_sentiment' to a Polars DataFrame based on sentiment analysis of the 'reddit_text' column.
    """
    texts = df['reddit_text'].to_list()
    sentiments = batch_sentiment(texts)  # Batch process all texts at once
    return df.with_columns(pl.Series("text_sentiment", sentiments))

