import polars as pl
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForTokenClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def get_sentiment(text: str) -> str:
    """Apply sentiment analysis to a given text and return the primary sentiment."""
    results = nlp(text)
    return results[0]['entity'] if results else 'null'

def fetch_texts_and_apply_sentiment(reply_ids, df):
    """Fetch texts for given reply IDs and apply sentiment analysis."""
    sentiments = []
    for reply_id in reply_ids:
        if reply_id is not None:
            # Fetch text for each reply_id
            text = df.filter(pl.col("reddit_name") == reply_id)["reddit_text"].to_list()
            if text:
                # Apply sentiment analysis
                sentiment = get_sentiment(text[0])
                sentiments.append(sentiment)
            else:
                sentiments.append('null')
        else:
            sentiments.append('null')
    return sentiments

def add_reply_sentiments(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a column called 'reply_sentiment' that contains the list of sentiments for each reply_id
    listed in the 'reply_ids' column of a dataframe. It uses a pre-loaded NLP model to determine
    the sentiment of the corresponding 'reddit_text'.
    """
    # Apply the custom function to each row
    result_df = df.with_columns(
        pl.col("reply_ids").apply(lambda ids: fetch_texts_and_apply_sentiment(ids, df)).alias("reply_sentiments")
    )

    return result_df