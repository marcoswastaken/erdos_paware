import polars as pl
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset
torch.cuda.empty_cache()

# Ensure the GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis").to(device)

# Set up the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)


def batch_sentiment(texts):
    """Process sentiment analysis in batches using datasets to utilize GPU more efficiently."""
    if not texts:
        return ['null'] * len(texts)  # Handle empty inputs

    # Trimming each text to 64 characters
    trimmed_texts = [text[:64] for text in texts]

    text_dataset = Dataset.from_dict({'text': trimmed_texts})

    def process_sentiment(batch):
        # The `batch['text']` is a list of texts
        processed_texts = nlp(batch['text'])
        return {'sentiment': [label_to_score(res['label']) for res in processed_texts]}

    # Batch processing
    sentiment_dataset = text_dataset.map(process_sentiment, batched=True, batch_size=8)
    return [item['sentiment'] for item in sentiment_dataset]

# Convert labels to numerical scores
def label_to_score(label):
    if label == 'negative' or 'NEGATIVE':
        return -1  # Negative
    elif label == 'neutral':
        return 0   # Neutral
    elif label == 'positive' or 'POSITIVE':
        return 1   # Positive

def add_text_and_summed_sentiments(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add 'text_sentiment' and 'summed_sentiments' columns to a DataFrame based on sentiment analysis of the 'reddit_text' column.
    """
    # Add 'text_sentiment' based on 'reddit_text'
    texts = df['reddit_text'].to_list()
    sentiments = batch_sentiment(texts)
    df = df.with_columns(pl.Series("text_sentiment", sentiments))

    # Calculate 'summed_sentiments' based on 'reply_ids' and new 'text_sentiment'
    summed_sentiments = []
    for reply_ids in df["reply_ids"]:
        if reply_ids is not None:
            # Find matching 'reddit_name' in 'reply_ids' and sum their 'text_sentiment'
            filter_mask = df["reddit_name"].is_in(reply_ids)
            filtered_df = df.filter(filter_mask)
            total_sentiment = filtered_df["text_sentiment"].sum()
            summed_sentiments.append(total_sentiment)
        else:
            summed_sentiments.append(0)  # If no reply_ids, sum is 0

    # Add the 'summed_sentiments' column
    df = df.with_columns(pl.Series("summed_sentiments", summed_sentiments))

    return df
